"""Build kernelfuse CUDA extension (Phase 8).

  pip install -e . --no-build-isolation

Requires torch with CUDA matching the local toolkit.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "kernelfuse" / "csrc"

setup(
    name="kernelfuse",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="kernelfuse._C",
            sources=[str(CSRC / "fused_add_rms_norm_op.cu")],
            include_dirs=[str(CSRC)],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
