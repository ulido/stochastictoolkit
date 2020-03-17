from setuptools import setup
from setuptools.extension import Extension
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ModuleNotFoundError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'
extensions = [Extension('stochastictoolkit._PDE', ['stochastictoolkit/_PDE' + ext])]
if USE_CYTHON:
    extensions = cythonize(extensions)

setup(name='stochastictoolkit',
      version='0.1',
      description='An ever expanding toolkit to build stochastic simulations in python',
      url='http://github.com/ulido/stochastictoolkit',
      author='Ulrich Dobramysl',
      author_email='ulrich.dobramysl@gmail.com',
      license='MIT',
      packages=['stochastictoolkit'],
      ext_modules = extensions,
      install_requires=[
          'cython',
          'numpy',
          'randomgen',
          'tqdm',
          'pandas',
          'shapely',
      ],
      test_suite='pytest',
      zip_safe=False)
