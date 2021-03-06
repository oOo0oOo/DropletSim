from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("c_code", ["c_code.pyx"], 
	include_dirs = [numpy.get_include()],
	extra_compile_args=['-fopenmp'],
	extra_link_args=['-fopenmp'])

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})