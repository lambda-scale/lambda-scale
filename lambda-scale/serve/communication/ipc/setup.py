from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ipc_p2p',
    ext_modules=[
        CUDAExtension(
        name='ipc_p2p', 
        sources=[   
                    'ipc_p2p.cpp', 
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
