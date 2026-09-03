"""Resource accounting that separates this backend from everything else.

On a 6 GB card shared with a simulator, "GPU at 74%" is a useless number: it
does not say whether the perception stack or the renderer is the thing filling
the card, and those have opposite remedies. So everything here is attributed.

WHAT CAN AND CANNOT BE MEASURED PER PROCESS, HONESTLY

  our GPU memory   EXACT. torch.cuda.memory_allocated is a byte count of this
                   process's own tensors, free to read and always right.
  other GPU memory INFERRED, as total minus ours. nvidia-smi exposes
                   per-process memory only under the TCC driver model; on a
                   consumer Windows card in WDDM it returns [N/A], verified on
                   this machine. Attributing the remainder to the simulator is
                   correct when the simulator is the only other GPU client, and
                   the field is named `other_mb` rather than `carla_mb` because
                   a browser with hardware acceleration is also in there.
  GPU utilisation  WHOLE CARD ONLY, for the same reason. Reported once, not
                   split, because a split would be invented.
  CPU per process  EXACT, both of them. GetProcessTimes is a Win32 call giving
                   kernel plus user time for any pid, so this backend and the
                   simulator are each measured directly rather than differenced.

Nothing here samples nvidia-smi on the request path. It costs 20-80 ms a call,
which is more than a whole perception frame, so it runs on its own thread and
readers take the last value. A telemetry system that slows the thing it
measures is reporting its own overhead.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time

# Processes worth naming separately in the dashboard. Matched case-insensitively
# on the executable name. CARLA ships as CarlaUE4 on both platforms; the plain
# "carla" catches a launcher script or a renamed build.
SIM_NAMES = ("carlaue4.exe", "carlaue4", "carla.exe", "carla")

_k32 = ctypes.windll.kernel32 if os.name == "nt" else None
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


def _cpu_seconds(pid: int):
    """Kernel + user CPU seconds for a pid, or None if it cannot be read.

    None is a real answer and is kept distinct from 0.0: a process that has not
    started and a process that is idle are different states, and showing an
    idle simulator as "0% CPU" when it is not running at all would be a lie the
    dashboard repeats every 200 ms.
    """
    if _k32 is None:
        try:
            import resource
            if pid == os.getpid():
                r = resource.getrusage(resource.RUSAGE_SELF)
                return r.ru_utime + r.ru_stime
        except Exception:
            pass
        return None
    h = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    c, e, kt, ut = _FT(), _FT(), _FT(), _FT()
    ok = _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                              ctypes.byref(kt), ctypes.byref(ut))
    _k32.CloseHandle(h)
    if not ok:
        return None
    q = lambda f: ((f.hi << 32) | f.lo) / 1e7      # noqa: E731  100 ns ticks
    return q(kt) + q(ut)


def find_pid(names=SIM_NAMES):
    """First pid whose image name matches, or None.

    One subprocess call, and the caller is expected to cache the answer: a pid
    does not change while a process lives, and enumerating 300 processes every
    frame to learn something that changes once an hour is waste.
    """
    try:
        r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=8)
    except Exception:
        return None
    for line in r.stdout.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        if parts[0].lower() in names:
            try:
                return int(parts[1])
            except ValueError:
                pass
    return None


def _smi(fields):
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if r.returncode:
            return None
        return [s.strip() for s in r.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None


class Telemetry:
    """Background sampler. Start once, read `snapshot()` as often as you like.

    The read is a dict copy of numbers already computed, so a 20 Hz dashboard
    costs nothing. Only the sampler thread ever touches nvidia-smi.
    """

    GPU_FIELDS = ("utilization.gpu", "memory.used", "memory.total",
                  "power.draw", "temperature.gpu", "clocks.current.graphics")

    def __init__(self, period=0.5, sim_names=SIM_NAMES):
        self.period = period
        self.sim_names = sim_names
        self.pid = os.getpid()
        self.sim_pid = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = None
        self._pt = None
        self._last = {}
        self._cpu = {}            # pid -> (cpu_seconds, wall)
        self.cores = os.cpu_count() or 1
        self.gpu_name = None
        self.gpu_total_mb = 0
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_total_mb = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e6)
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------- #
    def start(self):
        # Prime the CPU baseline here, not on the first loop iteration. A delta
        # needs two samples, so taking the first one inside the loop means the
        # first reading is None and the dashboard shows a dash for CPU until
        # the second tick. Priming makes the very first published sample real.
        self._cores_since(self.pid)
        if self._t is None:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
            # Process discovery gets its OWN thread. `tasklist` walks 300-odd
            # processes and takes one to two seconds, and anywhere on the
            # sampling loop it stalls every field for that long: the first
            # published CPU reading was None because the loop's first
            # iteration had not finished. Nothing that slow belongs on a
            # 500 ms cadence.
            self._pt = threading.Thread(target=self._pid_loop, daemon=True)
            self._pt.start()
        return self

    def stop(self):
        self._stop.set()
        for t in (self._t, getattr(self, "_pt", None)):
            if t:
                t.join(timeout=3)

    # -- sampling ----------------------------------------------------- #
    def _cores_since(self, pid):
        """Mean cores used by `pid` since the previous call for that pid."""
        now = time.perf_counter()
        cs = _cpu_seconds(pid)
        if cs is None:
            self._cpu.pop(pid, None)
            return None
        prev = self._cpu.get(pid)
        if prev is None:
            self._cpu[pid] = (cs, now)
            return None                    # first sample establishes a base
        dc, dw = cs - prev[0], now - prev[1]
        # Windows updates process CPU time on the scheduler tick, about every
        # 15.6 ms, NOT continuously. Divide a full tick by a 2 ms window and
        # the answer is "7.8 cores"; it reported 116 of 12 once, which is not
        # a number a machine can produce. So a sample is only published once
        # enough wall time has passed for the quantisation to be noise, and
        # until then the previous baseline is KEPT rather than replaced, so the
        # window widens instead of restarting.
        if dw < 0.25:
            return None
        self._cpu[pid] = (cs, now)
        return round(max(dc, 0.0) / dw, 3)

    def _loop(self):
        n = 0
        while not self._stop.is_set():
            out = {}
            # Our own numbers FIRST, before anything slow. `tasklist` walks
            # 300-odd processes and takes a second or two, and having it at the
            # top of the loop meant the first published snapshot was two
            # seconds late for every field, not just the one that needed it.
            out["cores_total"] = self.cores
            out["cpu_backend"] = self._cores_since(self.pid)
            out["cpu_sim"] = (self._cores_since(self.sim_pid)
                              if self.sim_pid else None)
            out["sim_running"] = self.sim_pid is not None

            n += 1

            g = _smi(self.GPU_FIELDS)
            if g:
                def f(i, d=0.0):
                    try:
                        return float(g[i])
                    except (ValueError, IndexError):
                        return d
                out["gpu_util"] = f(0)
                out["gpu_used_mb"] = f(1)
                out["gpu_total_mb"] = f(2) or self.gpu_total_mb
                out["gpu_power_w"] = f(3, float("nan"))
                out["gpu_temp_c"] = f(4)
                out["gpu_clock_mhz"] = f(5)

            try:
                import torch
                if torch.cuda.is_available():
                    out["gpu_backend_mb"] = round(
                        torch.cuda.memory_allocated() / 1e6, 1)
                    out["gpu_backend_peak_mb"] = round(
                        torch.cuda.max_memory_allocated() / 1e6, 1)
                    # what the caching allocator holds, which is what actually
                    # denies the simulator memory -- allocated() is only the
                    # part currently in tensors
                    out["gpu_backend_reserved_mb"] = round(
                        torch.cuda.memory_reserved() / 1e6, 1)
            except Exception:
                pass

            if "gpu_used_mb" in out and "gpu_backend_reserved_mb" in out:
                out["gpu_other_mb"] = round(
                    max(out["gpu_used_mb"] - out["gpu_backend_reserved_mb"], 0), 1)

            out["gpu_name"] = self.gpu_name
            out["t"] = time.time()
            with self._lock:
                self._last = out
            self._stop.wait(self.period)

    def _pid_loop(self):
        """Find the simulator, off the sampling path.

        A pid changes when a program starts or stops, so ten seconds is
        generous. Sleeping in small steps rather than one long wait so `stop()`
        does not have to wait out the interval.
        """
        while not self._stop.is_set():
            q = find_pid(self.sim_names)
            if q != self.sim_pid:
                self._cpu.pop(self.sim_pid, None)
                if q:
                    self._cores_since(q)          # prime the new one
                self.sim_pid = q
            for _ in range(20):
                if self._stop.is_set():
                    break
                self._stop.wait(0.5)

    def snapshot(self):
        with self._lock:
            return dict(self._last)
