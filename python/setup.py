import os
import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    README = fh.read()

# Find lib name
SOURCE_PATH = pathlib.Path(__file__).resolve()
LIB_NAME = None

for entry in os.listdir(SOURCE_PATH.parent / "pycubool"):
    if "cubool" in str(entry):
        LIB_NAME = entry
        break

setuptools.setup(
    name="pycubool",
    version="1.2.0",
    author="Egor Orachyov",
    author_email="egororachyov@gmail.com",
    license="MIT",
    description="cuBool library python bindings",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/JetBrains-Research/cuBool",
    project_urls={
        "cuBool project": "https://github.com/JetBrains-Research/cuBool",
        "Bug Tracker": "https://github.com/JetBrains-Research/cuBool/issues"
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    keywords=[
        "python",
        "cplusplus",
        "sparse-matrix",
        "linear-algebra",
        "graph-analysis",
        "graph-algorithms",
        "graphblas",
        "nvidia-cuda"
    ],
    packages=["pycubool"],
    package_dir={'': '.'},
    package_data={'': [LIB_NAME]},
    python_requires=">=3.0",
    include_package_data=True
)
