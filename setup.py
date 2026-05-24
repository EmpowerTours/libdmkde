"""setup.py — builds the dmkde._core C++ extension via pybind11.

Compile flags are split between MSVC and GCC/Clang because the two
toolchains do not share flag syntax. Pybind11Extension picks the
right C++ standard via `cxx_std`.
"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/W3", "/EHsc"]
else:
    extra_compile_args = ["-O3", "-Wall", "-Wextra"]

ext_modules = [
    Pybind11Extension(
        "dmkde._core",
        sources=["python/bindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
