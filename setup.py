from setuptools import setup

setup(name='stochastictoolkit',
      version='0.1',
      description='An ever expanding toolkit to build stochastic simulations in python',
      url='http://github.com/ulido/stochastictoolkit',
      author='Ulrich Dobramysl',
      author_email='ulrich.dobramysl@gmail.com',
      license='MIT',
      packages=['stochastictoolkit'],
      install_requires=[
          'numpy',
          'randomgen',
          'tqdm',
          'pandas',
          'shapely',
      ],
      zip_safe=False)
