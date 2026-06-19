from setuptools import setup, Extension
import pybind11
import numpy as np

ext_modules = [
    Extension(
        "polyinterp",
        ["polyinterp.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="polyinterp",
    ext_modules=ext_modules,
)
