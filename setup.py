"""
Setup script for Fused Kernel Library Python package.

This uses scikit-build-core to integrate CMake with Python packaging.
"""

from skbuild import setup
from setuptools import find_packages
import os

# Read version from pyproject.toml or use default
__version__ = "0.2.0"

# Read long description from README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="fused-kernel-library",
    version=__version__,
    description="Fused Kernel Library: Automatic GPU kernel fusion with Python bindings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Oscar Amoros Huguet",
    author_email="oscar.amoros.huguet@upc.edu",
    url="https://github.com/Libraries-Openly-Fused/FusedKernelLibrary",
    license="Apache-2.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "tvm-ffi>=0.1.0",
        "dlpack>=0.1.0",
    ],
    cmake_install_dir="python/fkl_ffi",
    cmake_args=[
        "-DBUILD_PYTHON_BINDINGS=ON",
        "-DENABLE_JIT=ON",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
)

