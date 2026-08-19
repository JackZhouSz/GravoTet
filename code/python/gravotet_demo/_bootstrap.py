"""Stdlib-only dependency bootstrap for the GravoTet supplementary demos.

The package ``__init__`` imports this module and runs it *before* importing any
submodule that pulls in numpy / scipy / matplotlib / pyvista / PIL at module
load time.  Because of that ordering this module must itself depend on nothing
outside the standard library, so it can run on a clean Python interpreter on
Windows, macOS, or Linux and pip-install whatever is missing.

Two properties matter more than anything else here, because this code runs
first on a reviewer's machine and a failure at this point means nothing else
gets a chance to work:

* A failure to install an *optional* package must never stop the demo.  Only
  the numerics and the residual/timing plot are required to produce the figure;
  the pyvista/VTK hierarchy renders are a bonus and are skipped when the wheel
  for the current interpreter does not exist.
* A failure to install a *required* package must produce an actionable message
  rather than a raw ``CalledProcessError`` traceback.  The common causes are
  PEP 668 externally-managed interpreters, a read-only ``site-packages``, and
  no network.
"""

from __future__ import annotations

import importlib
import os
import platform
import site
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent

# Lower bound is a hard requirement (the code uses 3.8 syntax and pybind11
# wheels start there).  The upper bound is advisory only: a newer interpreter
# is reported as untested and the run continues, because refusing to start on
# whatever python3 a vanilla 2026 OS ships is worse than a warning.
MIN_PY = (3, 8)
MAX_TESTED_PY = (3, 14)

# Import name -> pip requirement.  ``PIL`` is imported under that name but ships
# as the ``pillow`` distribution.  ``setuptools`` is needed to *build* the C++
# extension: ``setup.py`` and ``pybind11.setup_helpers`` import it, and on
# Python 3.12+ it also supplies the ``distutils`` shim the stdlib dropped.
REQUIRED = {
    "numpy": "numpy>=1.20",
    "scipy": "scipy>=1.7",
    "matplotlib": "matplotlib>=3.5",
    "pybind11": "pybind11>=2.10",
    "PIL": "pillow>=9.0",
    "setuptools": "setuptools>=60",
}

# Rendering only.  pyvista pulls in VTK, which is a large binary wheel that is
# routinely unavailable for the newest CPython or for less common OS/arch
# combinations.  When it is missing the demo still builds, still solves, and
# still writes output.png -- just without the hierarchy strip.
OPTIONAL = {
    "pyvista": "pyvista>=0.44",
}

_REEXEC_FLAG = "GRAVOTET_BOOTSTRAP_REEXEC"

_VENV_HINT = (
    "Create and activate a virtual environment, then re-run:\n"
    "    python3 -m venv .venv\n"
    "    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate\n"
    "    python -m pip install -r code/python/gravotet_demo/requirements.txt"
)


def ensure_python() -> None:
    """Reject interpreters below :data:`MIN_PY`; warn above :data:`MAX_TESTED_PY`."""
    v = sys.version_info[:2]
    if v < MIN_PY:
        raise SystemExit(
            f"GravoTet needs Python {MIN_PY[0]}.{MIN_PY[1]} or newer "
            f"(detected {v[0]}.{v[1]})."
        )
    if v > MAX_TESTED_PY:
        print(
            f"Note: Python {v[0]}.{v[1]} is newer than the tested range "
            f"({MIN_PY[0]}.{MIN_PY[1]}-{MAX_TESTED_PY[0]}.{MAX_TESTED_PY[1]}); "
            f"continuing.  If a dependency has no wheel for this version, use a "
            f"Python {MAX_TESTED_PY[0]}.{MAX_TESTED_PY[1]} virtual environment.",
            flush=True,
        )


def _missing(modules: dict) -> list:
    """Return the pip requirements for every module in `modules` that is absent."""
    out = []
    for module, spec in modules.items():
        try:
            importlib.import_module(module)
        except ImportError:
            out.append(spec)
    return out


def _run_pip(extra: list, specs: list) -> tuple:
    """Run one ``pip install`` invocation.  Returns ``(ok, combined_output)``."""
    cmd = [sys.executable, "-m", "pip", "install",
           "--disable-pip-version-check"] + extra + specs
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError as exc:  # pip is not installed at all
        return False, f"could not run pip: {exc}"
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


def _pip_install(specs: list) -> tuple:
    """Install `specs` with the running interpreter's pip.

    Falls back to a ``--user`` install when the interpreter's ``site-packages``
    is not writable or is a PEP 668 externally-managed environment (Debian /
    Ubuntu / Fedora system Python, Homebrew Python).  Returns ``(ok, log)`` and
    never raises.
    """
    ok, log = _run_pip([], specs)
    if ok:
        return True, ""

    needs_user_site = (
        "externally-managed-environment" in log
        or "Permission denied" in log
        or "Could not install packages due to an OSError" in log
    )
    if not needs_user_site:
        return False, log

    ok_user, log_user = _run_pip(["--user"], specs)
    if ok_user:
        return True, ""
    # Inside a venv, --user fails with its own message; the first log is the
    # one that actually explains the problem.
    return False, log if "not visible in this virtualenv" in log_user else log_user


def _refresh_import_paths() -> None:
    """Make freshly installed packages importable in the running interpreter."""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        site.addsitedir(user_site)
    importlib.invalidate_caches()


def _reexec_if_needed(modules: dict) -> None:
    """Restart the interpreter once if installed packages are still unimportable.

    pip can land a package in a ``site-packages`` directory that did not exist
    when the interpreter started (most often the per-user site directory).  No
    amount of cache invalidation makes that importable in-process on every
    platform, so re-exec once.  The environment flag makes a loop impossible.
    """
    if os.environ.get(_REEXEC_FLAG):
        return
    if not _missing(modules):
        return
    print("Restarting to pick up the newly installed packages...", flush=True)
    os.environ[_REEXEC_FLAG] = "1"
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except OSError:
        pass  # fall through; the import error below will be reported normally


def _optional_install_hint() -> str:
    """Explain a failed pyvista/VTK install and, where possible, how to fix it.

    VTK ships prebuilt binary wheels per (Python version, OS, CPU); pyvista
    itself is pure Python and always installs.  The gap a reviewer is most
    likely to hit is Apple Silicon on Python 3.8: no arm64 cp38 VTK wheel has
    ever been published, so no pyvista version can resolve there.  Saying so is
    worth a few lines, because otherwise the renders look permanently broken
    when they are one interpreter upgrade away.
    """
    base = (
        "  pyvista/VTK could not be installed on this interpreter.\n"
        "  Continuing without it: the demo still solves and still writes\n"
        "  output.png, but without the mesh-hierarchy strip."
    )
    on_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if sys.version_info[:2] < (3, 9) and on_apple_silicon:
        return base + (
            "\n  Reason: VTK publishes no Apple Silicon wheel for Python 3.8.\n"
            "  Re-run under Python 3.9 or newer to get the renders as well."
        )
    return base + (
        "\n  VTK only publishes wheels for selected interpreters; Python 3.9\n"
        "  or newer has wheels on every platform if you want the renders."
    )


def ensure_dependencies() -> None:
    """Install missing third-party packages into the running interpreter.

    Required and optional packages are installed in separate pip invocations so
    that a missing VTK wheel cannot take the whole run down with it.
    """
    missing_required = _missing(REQUIRED)
    if missing_required:
        print(
            f"Installing required packages: {', '.join(missing_required)}",
            flush=True,
        )
        ok, log = _pip_install(missing_required)
        if not ok:
            sys.stdout.flush()
            raise SystemExit(
                "Could not install the required Python packages.\n\n"
                f"{log.strip()[-2000:]}\n\n"
                f"{_VENV_HINT}"
            )
        _refresh_import_paths()

    # Anything installed in this run has to be importable afterwards, optional
    # packages included: a wheel that landed in a site directory the
    # interpreter did not have at startup is invisible until it restarts.
    installed_now = dict(REQUIRED) if missing_required else {}

    missing_optional = _missing(OPTIONAL)
    if missing_optional:
        print(
            f"Installing optional rendering packages: {', '.join(missing_optional)}",
            flush=True,
        )
        ok, _log = _pip_install(missing_optional)
        if ok:
            installed_now.update(OPTIONAL)
        else:
            print(_optional_install_hint(), flush=True)
        _refresh_import_paths()

    if installed_now:
        _reexec_if_needed(installed_now)
