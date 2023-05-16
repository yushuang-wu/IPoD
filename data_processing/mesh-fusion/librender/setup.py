from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extra_compile_args = ["-ffast-math", '-msse', '-msse2', '-msse3', '-msse4.2', '-O4', '-fopenmp']
extra_link_args = ['-lGLEW', '-lglut', '-lGL', '-lGLU', '-fopenmp']

ext_modules = [
    Extension('pyrender',
      [
        'pyrender.pyx',
        'offscreen.cpp',
      ],
      language='c++',
      include_dirs=[np.get_include()],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args
    )
  ]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3
setup(
  name="pyrender",
  cmdclass= {'build_ext': build_ext},
  ext_modules=ext_modules
)


