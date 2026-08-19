"""Build the supplementary `gravotet` C++ extension.

Invoked from `code/python/gravotet_demo/`; C++ sources live under `code/cpp/`
with vendored Eigen headers under `code/deps/eigen/`. No SuiteSparse/CHOLMOD
dependency.
"""

from __future__ import annotations

import platform
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).resolve().parent  # code/python/gravotet_demo/
CODE = HERE.parent.parent               # code/
SRC = CODE / "cpp"
EIGEN = CODE / "deps" / "eigen"

if platform.system() == "Windows":
    # Two MSVC workarounds for recent VS 2022 (17.14 / MSVC 14.44) toolsets:
    #   /GL- disables the LTCG whole-program optimisation injected by the Python
    #   build machinery, which triggers an ICE on Eigen sparse templates.
    #   /Ob1 caps inlining at one level. The default /O2 (implicit /Ob2) triggers
    #   a separate internal compiler error (C1001) during <filesystem>/<execution>
    #   and Eigen code generation on MSVC 14.44.35207. /Ob1 avoids it and retains
    #   /O2 speed. Both flags are harmless on toolsets without the bug.
    extra_compile_args = ["/O2", "/Ob1", "/EHsc", "/bigobj", "/GL-"]
    extra_link_args: list[str] = []
else:
    # No macOS deployment-target override is needed: the C++ sources use no
    # library feature that Apple's libc++ marks as availability-limited.  The
    # <filesystem> dependency that used to force -mmacosx-version-min=10.15 was
    # replaced by a plain mkdir() helper in multigrid_solver.cpp, so the
    # interpreter's own deployment target is always sufficient.  Passing the
    # flag anyway put two conflicting -mmacosx-version-min values on the link
    # line, which is the kind of thing that breaks on the next toolchain.
    extra_compile_args = ["-O3", "-fPIC"]
    extra_link_args = []

PACKAGE_VERSION = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "gravotet",
        [
            str(SRC / "gravotet_binding.cpp"),
            str(SRC / "multigrid_solver.cpp"),
            str(SRC / "multigrid_solver_vcycle.cpp"),
        ],
        include_dirs=[str(SRC), str(EIGEN)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("VERSION_INFO", PACKAGE_VERSION)],
        cxx_std=17,
        language="c++",
    )
]

setup(
    name="gravotet",
    version=PACKAGE_VERSION,
    description="Minimal GravoTet MG supplementary bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
