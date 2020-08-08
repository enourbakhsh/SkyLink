from setuptools import setup
from .version import __version__

setup(name='skylink',
      version=__version__,
      description='Using KDTree search and graphs to match sky catalogs.',
      url='https://github.com/skylink',
      author='Erfan Nourbakhsh',
      author_email='erfanyz@gmail.com',
      license='MIT',
      packages=['skylink'],
      install_requires=[
          "numpy",
          "astropy",
          "networkx", 
          "networkit", 
          "igraph", 
          "pandas", #pandas>=0.25.3
          "busypal @ git+https://github.com/enourbakhsh/busypal", 
          "colored", 
          "tqdm",
      ],
#       packages=[
#           "numpy",
#           "astropy",
#           "networkx", 
#           "networkit", 
#           "igraph", 
#           "pandas", #pandas>=0.25.3
#           "busypal @ git+https://github.com/enourbakhsh/busypal", 
#           "colored", 
#           "tqdm",
#       ],
      zip_safe=False)
