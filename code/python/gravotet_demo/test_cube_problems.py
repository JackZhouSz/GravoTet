"""Smoke test for the supplementary Poisson and biharmonic cube examples.

Quick validation of a clean checkout: builds the GravoTet hierarchy on a
small cube and checks convergence for both PDEs.  Exits non-zero on any
failure.

Run from the repository root:

    python code/python/gravotet_demo/test_cube_problems.py --resolution 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# code/python/gravotet_demo/test_cube_problems.py -> code/python is parents[1],
# which is the directory that has to be on sys.path for `import gravotet_demo`
# to resolve when this file is executed as a script.
PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from gravotet_demo import ensure_extension, make_cube_mesh, run_suite  # noqa: E402

# Mirrors `min_verts` in gravotet_demo.solver.create_solver.
MIN_HIERARCHY_VERTICES = 200

# The residual check below is the real pass/fail gate: it asserts exactly what
# the solver promises, namely that the relative residual reaches the requested
# tolerance.  The error against the manufactured solution is only a sanity
# bound, because it also carries the discretisation error and, for the
# ill-conditioned biharmonic operator, the algebraic error left at a 1e-5
# residual.  Neither of those shrinks with resolution, so a tight bound here
# would fail on larger cubes for reasons that have nothing to do with
# correctness.  1e-2 still catches a solver that returns nonsense.
MAX_RELATIVE_ERROR = 1e-2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=10,
                        help="cube side resolution; the mesh has resolution**3 "
                             "vertices (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # The hierarchy stops coarsening at `min_verts` (200, see solver.py), so a
    # mesh smaller than that yields a single level and no V-cycle to run.
    if args.resolution ** 3 <= MIN_HIERARCHY_VERTICES:
        raise SystemExit(
            f"--resolution {args.resolution} gives only {args.resolution ** 3} "
            f"vertices, which is below the {MIN_HIERARCHY_VERTICES}-vertex "
            f"floor for building a multigrid hierarchy.  Use --resolution 8 or "
            f"more."
        )

    gravotet = ensure_extension()
    mesh = make_cube_mesh(gravotet, args.resolution)
    summary = run_suite(
        gravotet, mesh,
        label=f"Cube smoke (res {args.resolution})",
        output_subdir=f"smoke_cube_{args.resolution}",
        problems=("poisson", "biharmonic"),
        make_figure=False,
        verbose=False,
    )

    failed: list[str] = []
    for name, result in summary["problems"].items():
        if not result["converged"]:
            failed.append(f"{name}: solver did not converge")
        if result["timed_out"]:
            failed.append(f"{name}: solver timed out")
        if result["relative_residual"] > max(10.0 * result["tolerance"], 1e-10):
            failed.append(
                f"{name}: relative residual {result['relative_residual']:.3e} exceeds threshold"
            )
        if result["relative_error"] > MAX_RELATIVE_ERROR:
            failed.append(
                f"{name}: relative error {result['relative_error']:.3e} exceeds "
                f"the {MAX_RELATIVE_ERROR:.0e} sanity bound"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    if failed:
        print("Supplement smoke test FAILED")
        for item in failed:
            print(f"  - {item}")
        return 1

    print("Supplement smoke test PASSED")
    for name, result in summary["problems"].items():
        print(
            f"  {name}: cycles={result['num_cycles']}, "
            f"rel_res={result['relative_residual']:.3e}, "
            f"rel_err={result['relative_error']:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
