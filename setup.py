from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cellmap_segmentation_challenge.utils.rand_voi",
        sources=["src/cellmap_segmentation_challenge/utils/rand_voi.pyx"],
        include_dirs=[
            np.get_include(),
            "src/cellmap_segmentation_challenge/utils",
        ],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={"embedsignature": True},
    )
)
