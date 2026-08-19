"""GravoTet supplementary demo: cube at ~200k vertices.

Identical to the sphere and torus drivers in everything except the input
mesh: solves Poisson and biharmonic on a regular cube with 59**3 = 205,379
vertices.  All shared logic lives in ``gravotet_demo.run_demo``.

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

from gravotet_demo import make_cube_mesh, run_demo  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_demo(
        label="Cube 200k",
        output_subdir="cube_200k",
        build_mesh=lambda gravotet: make_cube_mesh(gravotet, 59),
        description=__doc__,
    ))
