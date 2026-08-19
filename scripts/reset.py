"""Reset the GravoTet supplementary demo to a clean state.

Removes all runtime-generated files: output figures, JSON summaries, the
compiled C++ extension, and setuptools build artifacts. The source tree and
vendored deps are left untouched.

Run from the repository root or from this directory:

    python scripts/reset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "code" / "python" / "gravotet_demo"

# Directories to remove entirely.  `output/` is deliberately NOT here: it holds
# the four committed reference figures that README.md embeds, so removing it
# would delete tracked results.  It is cleaned selectively in `_clean_output()`.
_DIRS_TO_REMOVE = [
    PKG / "build",
    PKG / "__pycache__",
]

# Per-demo artifacts a run generates inside output/<demo>/.  Everything else
# there, in particular the tracked `output.png`, is preserved.
_OUTPUT_DIRS = ("meshes",)
_OUTPUT_FILES = (
    "data.json",
    "prolongations.npz",
    "combined.png",
    "hierarchy_strip.png",
)

# Glob patterns for individual files to remove inside gravotet_demo/
_FILE_GLOBS = [
    "gravotet*.pyd",   # Windows compiled extension
    "gravotet*.so",    # Linux / macOS compiled extension
    "gravotet*.dylib", # macOS dylib (if any)
    "*.egg-info",      # setuptools metadata directories
]


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"  removed  {path.relative_to(ROOT)}")
    else:
        print(f"  skipped  {path.relative_to(ROOT)}  (not found)")


def _remove_file(path: Path) -> None:
    if path.is_file():
        path.unlink()
        print(f"  removed  {path.relative_to(ROOT)}")
    elif path.is_dir():  # egg-info shows up as a directory
        shutil.rmtree(path)
        print(f"  removed  {path.relative_to(ROOT)}")


def _clean_output() -> None:
    """Remove generated artifacts under output/, keeping the tracked figures."""
    output = ROOT / "output"
    if not output.is_dir():
        return
    for demo_dir in sorted(p for p in output.iterdir() if p.is_dir()):
        for name in _OUTPUT_DIRS:
            if (demo_dir / name).is_dir():
                _remove_dir(demo_dir / name)
        for name in _OUTPUT_FILES:
            _remove_file(demo_dir / name)


def main() -> None:
    print("Resetting GravoTet supplementary demo...\n")

    print("Generated output (committed reference figures are kept):")
    _clean_output()

    print("\nDirectories:")
    for d in _DIRS_TO_REMOVE:
        _remove_dir(d)

    print("\nBuild artifacts in code/python/gravotet_demo/:")
    found_any = False
    for pattern in _FILE_GLOBS:
        for match in PKG.glob(pattern):
            _remove_file(match)
            found_any = True
    if not found_any:
        print("  skipped  (none found)")

    print("\nDone. Repository is in a clean state.")
    print("Run  python run_demo_cube_200k.py  to rebuild and re-run a demo.")


if __name__ == "__main__":
    main()
