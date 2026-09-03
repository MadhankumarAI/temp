"""Put the 2.5D grid engine on the import path.

`grid25.py` and `kitti.py` are the teammate's, and they live in
mapping/Lidar-2.5d-mapping rather than beside this package. Everything here
imports them plainly as `grid25` and `kitti`, which only resolves if that
directory is on sys.path.

Until now it always was by accident: the work was done from a scratch copy of
that repository where the modules sat alongside. Run from the project root the
same commands fail with ModuleNotFoundError, which is a poor thing to hand
someone along with a list of commands.

Import this first, before grid25:

    import _bootstrap        # noqa: F401
    import grid25 as g

Deliberately not a package-relative import: these modules are also copied next
to grid25.py and run there, and this has to work in both places, so it adds the
path only when it is missing.
"""

import pathlib
import sys

_MAPPING = (pathlib.Path(__file__).resolve().parents[1]
            / "mapping" / "Lidar-2.5d-mapping")

if _MAPPING.is_dir() and str(_MAPPING) not in sys.path:
    sys.path.insert(0, str(_MAPPING))

# also allow sibling imports when a module here is run as a script
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
