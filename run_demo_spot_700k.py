"""GravoTet supplementary demo: Spot at ~700k vertices.

Solves Poisson and biharmonic on the Spot tetrahedral mesh
(``data/spot.ply``, ~709k vertices).  All shared logic lives in
``gravotet_demo.run_demo``.

Usage:
    python run_demo_spot_700k.py
    python run_demo_spot_700k.py --problems poisson
    python run_demo_spot_700k.py --problems poisson biharmonic --verbose
    python run_demo_spot_700k.py --max-cycles 500

Options:
    --problems {poisson,biharmonic} [...]   PDE(s) to solve (default: both)
    --max-cycles N                          V-cycle iteration cap (default: 500)
    --verbose                               enable the C++ hierarchy log
"""

from __future__ import annotations

import sys
from pathlib import Path

# Answer --help without side effects.  Importing the package below runs the
# dependency bootstrap, which on a clean machine installs several hundred
# megabytes of wheels and then compiles the C++ extension; a reviewer asking
# what the flags are should not trigger that.
if {"-h", "--help"}.intersection(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "python"))

from gravotet_demo import DATA_DIR, make_mesh_from_ply, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Spot 700k",
        output_subdir="spot_700k",
        build_mesh=lambda gravotet: make_mesh_from_ply(
            gravotet, DATA_DIR / "spot.ply",
        ),
        description=__doc__,
    ))
