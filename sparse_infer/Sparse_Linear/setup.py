# setup.py

# https://chatgpt.com/c/68624ab5-0478-8010-965f-a22ade0693b2

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_linear',
    ext_modules=[
        CUDAExtension(
            name='sparse_linear',
            sources=['sparse_linear_impl.cu'],  # .cu 里是 kernel
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-lineinfo', '-arch=sm_75']  # 按需修改 sm 架构
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)