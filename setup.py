import os
from setuptools import setup

package_name = 'skylink'
lib_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), package_name)
exec(open(os.path.join(lib_path, 'version.py')).read()) # it reads __version__

setup(name=package_name,
      version=__version__,
      description='Using KDTree search and graphs to match sky catalogs.',
      url='https://github.com/skylink',
      author='Erfan Nourbakhsh',
      author_email='erfanyz@gmail.com',
      license='MIT',
      packages=[package_name, package_name+'.astropy_search'],
      package_data={"": ["*.rst"]},
      install_requires=[
          "numpy",
          "astropy",
          "networkx", 
          "networkit", 
          "python-igraph", 
          "pandas", #pandas>=0.25.3
          "busypal @ git+https://github.com/enourbakhsh/busypal", 
          "colored", 
          "tqdm",
      ],
      zip_safe=False)
