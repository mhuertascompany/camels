 from setuptools import setup

 setup(
   name='Camels',
   version='0.1.0',
   author='TFG Cecilia',
   author_email='mhuertas@iac.es',
   url='https://github.com/mhuertascompany/camels',
   packages=find_packages(),
   install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)