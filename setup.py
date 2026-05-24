"""setup.py — builds the dmkde._core C++ extension via pybind11."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "dmkde._core",
        sources=["python/bindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=["-O3", "-Wall", "-Wextra"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
