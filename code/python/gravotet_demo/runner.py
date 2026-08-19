"""Shared bootstrap and execution logic for the GravoTet supplementary demos.

The four driver scripts at the repository root (``run_demo_cube_200k.py``,
``run_demo_spot_700k.py``, ``run_demo_sphere_1.5M.py``, ``run_demo_torus_2M.py``)
all funnel through this module so that environment setup, extension build,
hierarchy construction, V-cycle solve, and on-disk export are written exactly
once.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from . import _bootstrap

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "code"
PKG = CODE / "python" / "gravotet_demo"
SRC = CODE / "cpp"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
# The Python-version and third-party-package checks live in the stdlib-only
# :mod:`_bootstrap` module, which the package __init__ runs before importing
# any submodule that needs those packages.  The extension build below stays
# here because it relies on the package already being importable.

def _ext_paths() -> list[Path]:
    return [PKG / f"gravotet{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]


def _needs_rebuild() -> bool:
    built = [p for p in _ext_paths() if p.exists()]
    if not built:
        return True
    ext_mtime = max(p.stat().st_mtime for p in built)
    sources = [
        PKG / "setup.py",
        SRC / "gravotet_binding.cpp",
        SRC / "multigrid_solver.h",
        SRC / "multigrid_solver.cpp",
        SRC / "multigrid_solver_vcycle.cpp",
    ]
    return any(p.stat().st_mtime > ext_mtime for p in sources)


# Recognisable compiler-not-installed messages, mapped to the command that
# fixes them.  Setuptools reports these from deep inside the build, where they
# read as an internal error rather than "install a compiler".
_COMPILER_HINTS = (
    ("Microsoft Visual C++",
     "Install the Visual Studio Build Tools and select the "
     "'Desktop development with C++' workload:\n"
     "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"),
    ("unable to execute", "Install a C++17 compiler (see the README Requirements section)."),
    ("No such file or directory: 'c++'",
     "Install a C++ toolchain:  sudo apt install build-essential python3-dev"),
    ("No such file or directory: 'gcc'",
     "Install a C++ toolchain:  sudo apt install build-essential python3-dev"),
    ("command 'gcc' failed",
     "Install a C++ toolchain:  sudo apt install build-essential python3-dev"),
    ("xcrun: error",
     "Install the Xcode command line tools:  xcode-select --install"),
    ("Python.h",
     "Install the Python development headers:  sudo apt install python3-dev"),
)

BUILD_LOG = PKG / "build.log"


def _build_extension() -> None:
    """Compile the pybind11 extension in place, writing a full log to disk."""
    print("Building the local GravoTet extension "
          "(one-time, typically 1-3 minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
        cwd=PKG,
        capture_output=True,
        text=True,
        errors="replace",
    )
    log = (result.stdout or "") + (result.stderr or "")
    try:
        BUILD_LOG.write_text(log, errors="replace")
    except OSError:
        pass

    if result.returncode != 0:
        print("\nExtension build FAILED. Full build log follows:\n", flush=True)
        print(log, end="" if log.endswith("\n") else "\n", flush=True)
        for needle, hint in _COMPILER_HINTS:
            if needle in log:
                print(f"\nLikely cause and fix:\n  {hint}\n", flush=True)
                break
        raise SystemExit(
            f"Could not build the GravoTet C++ extension. "
            f"The full log is at {BUILD_LOG}."
        )
    print(f"Extension build finished (log: {BUILD_LOG}).", flush=True)


def _line_buffer_stdout() -> None:
    """Keep Python's prints in order with the C++ solver's own logging.

    The extension writes to the C stdout while Python buffers its own, so when
    the output is piped or redirected the two streams interleave wrongly and
    the C++ progress log appears after the final summary.  Line buffering on
    the Python side restores the real ordering.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


def ensure_extension() -> Any:
    """Bootstrap Python deps, (re)build the C++ extension, return the module."""
    _line_buffer_stdout()
    _bootstrap.ensure_python()
    _bootstrap.ensure_dependencies()
    if str(PKG.parent) not in sys.path:
        sys.path.insert(0, str(PKG.parent))   # for the gravotet_demo package
    if str(PKG) not in sys.path:
        sys.path.insert(0, str(PKG))          # for the gravotet C++ extension

    if _needs_rebuild():
        _build_extension()
    importlib.invalidate_caches()

    try:
        return importlib.import_module("gravotet")
    except ImportError as exc:
        # A compiled extension left over from another interpreter, another
        # machine, or an interrupted build imports as an opaque ImportError.
        # Rebuilding from scratch is nearly always the right answer, so do it
        # once automatically rather than making the reviewer diagnose it.
        print(f"\nImporting the built extension failed ({exc}).", flush=True)
        print("Rebuilding it from scratch and retrying...\n", flush=True)
        for stale in _ext_paths():
            try:
                stale.unlink()
            except OSError:
                pass
        _build_extension()
        importlib.invalidate_caches()
        return importlib.import_module("gravotet")


# Artifacts a previous run of the same demo may have left behind.  Everything
# else in output/<demo>/ is preserved -- in particular the committed reference
# output.png that README.md embeds, which must survive a run that fails before
# it can write a replacement.
_GENERATED_DIRS = ("meshes",)
_GENERATED_FILES = (
    "data.json",
    "prolongations.npz",
    "combined.png",
    "hierarchy_strip.png",
)


def _prepare_output_dir(output_dir: Path) -> None:
    """Clear the artifacts of a previous run, leaving tracked files in place."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in _GENERATED_DIRS:
        target = output_dir / name
        if target.is_dir():
            shutil.rmtree(target)
    for name in _GENERATED_FILES:
        (output_dir / name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------

def build_hierarchy(gravotet: Any, mesh: Any, verbose: bool = False) -> tuple[Any, float]:
    """Construct the GravoTet hierarchy and return (solver, build_time_ms)."""
    from .solver import create_solver
    return create_solver(gravotet, mesh, verbose=verbose)


def _concat_hierarchy_below_plots(output_dir: Path) -> None:
    """Merge hierarchy_strip.png below combined.png into a single output.png.

    The hierarchy strip is resized to match the width of the combined figure
    and placed directly underneath it, producing a single ``output.png`` that
    contains both the residual/timing plots and the mesh hierarchy.
    Intermediate files (``combined.png`` and ``hierarchy_strip.png``) are
    deleted so only ``output.png`` remains.
    """
    from PIL import Image

    combined_path = output_dir / "combined.png"
    strip_path = output_dir / "hierarchy_strip.png"
    output_path = output_dir / "output.png"

    if not combined_path.exists():
        return

    # No hierarchy strip means pyvista/VTK was unavailable or a render failed.
    # output.png already holds the residual/timing figure (see run_suite), so
    # just drop the intermediate rather than leaving the demo with no figure.
    if not strip_path.exists():
        combined_path.unlink(missing_ok=True)
        return

    combined = Image.open(str(combined_path))
    strip = Image.open(str(strip_path))

    # Resize strip to match combined width, preserving aspect ratio
    target_w = combined.width
    scale = target_w / strip.width
    new_h = int(strip.height * scale)
    strip_resized = strip.resize((target_w, new_h), Image.LANCZOS)

    # Vertically concatenate: combined on top, hierarchy strip below
    total_h = combined.height + new_h
    result = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    result.paste(combined, (0, 0))
    result.paste(strip_resized, (0, combined.height))

    combined.close()
    strip.close()

    result.save(str(output_path), "PNG")

    # Clean up intermediates — only output.png should remain
    combined_path.unlink(missing_ok=True)
    strip_path.unlink(missing_ok=True)


def _guarded(what: str, fn: Callable, *args) -> None:
    """Run an output-writing step, reporting rather than propagating a failure.

    Everything these steps produce is secondary to the solve that already
    completed, so a matplotlib, VTK, or filesystem problem in one of them is
    logged and the run continues.
    """
    try:
        fn(*args)
    except Exception as exc:
        print(f"  [warn] {what} failed ({type(exc).__name__}: {exc}); continuing.",
              flush=True)


def _stage_output_figure(output_dir: Path) -> None:
    """Copy the residual/timing figure to output.png as an interim deliverable."""
    combined_path = output_dir / "combined.png"
    if combined_path.exists():
        shutil.copyfile(str(combined_path), str(output_dir / "output.png"))


def run_suite(
    gravotet: Any,
    mesh: Any,
    *,
    label: str,
    output_subdir: str,
    problems: Iterable[str] = ("poisson",),
    make_figure: bool = False,
    verbose: bool = False,
    max_cycles: int = 150,
) -> dict[str, Any]:
    """Build the hierarchy, solve every requested PDE, and write the outputs.

    Parameters
    ----------
    gravotet      : imported C++ extension module
    mesh          : ``gravotet.TetrahedralMesh``
    label         : human-readable label for log lines (e.g. ``"Cube 100k"``)
    output_subdir : name of the per-suite directory under ``output/``
    problems      : iterable of ``"poisson"`` and/or ``"biharmonic"``
    make_figure   : generate ``combined_<subdir>.png`` (residual + timing)
    verbose       : forward to the C++ solver
    """
    from .pde import assemble_problem
    from .solver import solve_problem
    from .export import save_hierarchy_meshes, save_prolongation_matrices

    print(f"\n{'='*60}")
    print(f"  {label}  ({int(mesh.num_vertices()):,} vertices, "
          f"{int(mesh.num_tetrahedra()):,} tets)")
    print(f"{'='*60}")

    solver, hierarchy_build_ms = build_hierarchy(gravotet, mesh, verbose=verbose)

    summary: dict[str, Any] = {
        "label": label,
        "binding_version": gravotet.version(),
        "num_vertices": int(mesh.num_vertices()),
        "num_tetrahedra": int(mesh.num_tetrahedra()),
        "hierarchy_levels": int(solver.num_levels()),
        "hierarchy_build_ms": float(hierarchy_build_ms),
        "hierarchy_level_sizes": [
            {
                "level": i,
                "vertices": int(len(solver.all_vertices[i])),
                "tetrahedra": int(len(solver.all_tetrahedra[i])),
            }
            for i in range(len(solver.all_vertices))
        ],
        "problems": {},
    }

    for name in problems:
        # Announce each phase separately: on the multi-million-vertex meshes
        # assembly and solve each take minutes, and a single line followed by
        # silence reads as a hang.
        t0 = time.perf_counter()
        print(f"  Assembling the {name} system "
              f"({int(mesh.num_vertices()):,} unknowns) ...", flush=True)
        problem = assemble_problem(solver, name, resolution=int(mesh.num_vertices()))
        problem.hierarchy_build_ms = hierarchy_build_ms
        print(f"  Solving {name} (up to {max_cycles} V-cycles, "
              f"tolerance {problem.tolerance:g}) ...", flush=True)
        result = solve_problem(solver, problem, max_cycles=max_cycles)
        result["assemble_and_solve_wall_s"] = time.perf_counter() - t0
        summary["problems"][name] = result

    output_dir = ROOT / "output" / output_subdir
    _prepare_output_dir(output_dir)

    # The metrics are the product of the expensive part of the run, so they go
    # to disk before anything that can fail on a rendering or export detail.
    (output_dir / "data.json").write_text(json.dumps(summary, indent=2))

    if make_figure:
        from .plotting import save_combined_figure
        _guarded("residual/timing figure", save_combined_figure, summary, output_dir)
        # Publish the figure as output.png right away.  VTK can abort the whole
        # process rather than raise when it cannot get a GL context, so the
        # deliverable must already be on disk before any rendering starts; the
        # concatenation step below simply replaces it with the richer version.
        _guarded("figure staging", _stage_output_figure, output_dir)

    _guarded("hierarchy mesh export", save_hierarchy_meshes, solver, output_dir)
    _guarded("prolongation export", save_prolongation_matrices, solver, output_dir)

    if make_figure:
        # Concatenate hierarchy strip below the combined residual+timing figure
        # so the supplementary produces a single self-contained image.
        _guarded("figure assembly", _concat_hierarchy_below_plots, output_dir)

    _print_summary(label, summary)
    print(f"  Output: {output_dir}")
    if not (output_dir / "output.png").exists() and make_figure:
        print("  Note: output.png was not produced; see the messages above.")
    return summary


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n--- {label} results ---")
    print(f"  Binding: {summary['binding_version']}")
    print(
        f"  Levels: {summary['hierarchy_levels']}  |  "
        f"hierarchy build: {summary['hierarchy_build_ms']:.1f} ms"
    )
    for lvl in summary["hierarchy_level_sizes"]:
        print(f"    Level {lvl['level']}: "
              f"{lvl['vertices']:,} verts, {lvl['tetrahedra']:,} tets")
    unconverged = []
    for name, result in summary["problems"].items():
        print(
            f"  [{name}] converged={result['converged']}  "
            f"cycles={result['num_cycles']}  "
            f"rel_res={result['relative_residual']:.3e}  "
            f"solve={result['total_time_ms']:.1f} ms"
        )
        if not result["converged"]:
            unconverged.append((name, result))

    # A run that stops on the cycle cap still writes a plausible-looking figure,
    # so say so unmistakably rather than letting the reader assume convergence.
    for name, result in unconverged:
        reason = ("hit the timeout" if result.get("timed_out")
                  else f"hit the {result['num_cycles']}-cycle cap")
        print(
            f"\n  WARNING: the {name} solve did not reach its tolerance "
            f"({result['tolerance']:.0e}); it {reason} at a relative residual "
            f"of {result['relative_residual']:.3e}.\n"
            f"  The figure and data.json below describe that partial solve.\n"
            f"  Re-run with a larger --max-cycles to continue."
        )


# ---------------------------------------------------------------------------
# Argparse helper used by the driver scripts
# ---------------------------------------------------------------------------

DEFAULT_PROBLEMS = ("poisson", "biharmonic")
DEFAULT_MAX_CYCLES = 500


def parse_common_args(description: str) -> argparse.Namespace:
    """Common CLI surface shared by every driver script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable C++ hierarchy logging")
    parser.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        choices=["poisson", "biharmonic"],
        help="PDE(s) to solve (default: %(default)s)",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=DEFAULT_MAX_CYCLES,
        help="V-cycle iteration cap (default: %(default)s)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Unified driver entry point
# ---------------------------------------------------------------------------

def run_demo(
    label: str,
    output_subdir: str,
    build_mesh: Callable[[Any], Any],
    description: str | None = None,
) -> int:
    """End-to-end demo entry shared by every driver script.

    A driver supplies (a) a human-readable ``label``, (b) the ``output_subdir``
    name under ``output/``, and (c) a ``build_mesh(gravotet) -> TetrahedralMesh``
    callable.  Everything else - environment bootstrap, extension build,
    hierarchy construction, Poisson + biharmonic solves, residual / timing
    figure, hierarchy mesh and prolongation export, JSON summary, log
    output - is identical across demos and lives here.
    """
    args = parse_common_args(description or label)
    gravotet = ensure_extension()
    mesh = build_mesh(gravotet)
    run_suite(
        gravotet, mesh,
        label=label,
        output_subdir=output_subdir,
        problems=args.problems,
        make_figure=True,
        verbose=args.verbose,
        max_cycles=args.max_cycles,
    )
    return 0
