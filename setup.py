from setuptools import find_packages
from setuptools import setup

setup(name='camels',
      description='camels package',
      author='AstroInfo',
      packages=find_packages(),
      install_requires=['astropy', 'tensorflow_datasets']
      )