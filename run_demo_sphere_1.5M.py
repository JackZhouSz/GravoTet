"""GravoTet supplementary demo: sphere at ~1.5M vertices.

Identical to the cube and torus drivers in everything except the input
mesh: solves Poisson and biharmonic on a TetGen-meshed unit ball
(``data/sphere_1.5M.ply``, ~1.45 M vertices).  All shared logic lives in
``gravotet_demo.run_demo``.

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
        label="Sphere 1.5M",
        output_subdir="sphere_1.5M",
        build_mesh=lambda gravotet: make_mesh_from_ply(
            gravotet, DATA_DIR / "sphere_1.5M.ply",
        ),
        description=__doc__,
    ))
