[![build](https://github.com/enourbakhsh/SkyLink/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/enourbakhsh/SkyLink/actions/workflows/build.yml)
[![flake8](https://github.com/enourbakhsh/SkyLink/actions/workflows/flake8.yml/badge.svg?branch=master)](https://github.com/enourbakhsh/SkyLink/actions/workflows/flake8.yml)
[![pytest](https://github.com/enourbakhsh/SkyLink/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/enourbakhsh/SkyLink/actions/workflows/pytest.yml)
[![GitHub license](https://badgen.net/github/license/enourbakhsh/SkyLink)](https://github.com/enourbakhsh/SkyLink/blob/master/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/enourbakhsh/SkyLink/graphs/commit-activity)
# SkyLink
Code for efficiently matching sky catalogs using KDTrees and graphs. This includes internal matching via friends-of-friends algorithm. Even in its serial mode, `SkyLink` performs faster than many other approaches I came across (e.g. the `FoFCatalogMatching`+`fast3tree` packages hosted [here](https://github.com/yymao/FoFCatalogMatching) and [here](https://bitbucket.org/yymao/fast3tree), respectively).
## Example
A minimal usage with three catalogs and multiple values for the linking length<sup>[1](#footnote1)</sup> looks something like the following. This has been made similar in use to the `FoFCatalogMatching` package to provide an effortless way to switch to this new package and speed up your matching codes.
Note that the range of the sky coordinates in the parallel mode should allow for a gnomonic projection which will be used in mosaicking.

``` python
from numpy.random import uniform, normal
from astropy.table import Table
import skylink as sl

n = 1000
ra = uniform(0, 179, n)
dec = uniform(-90, 90, n)

# Create catalogs
cat_a = Table({'ra': ra, 'dec': dec})
cat_b = Table({'ra': ra + normal(0, 0.0002, n),
               'dec': dec + normal(0, 0.0002, n)})
cat_c = Table({'ra': ra + normal(0, 0.0003, n),
               'dec': dec + normal(0, 0.0003, n)})

# Run an FoF match with 8 processors in parallel
res = sl.match({'a': cat_a, 'b':cat_b, 'c':cat_c},
               {3.0: 3, 2.0: 3, 1: None}, nprocs = 8)
```
<a name="footnote1"><sup>1</sup></a> *See [FoFCatalogMatching](https://github.com/yymao/FoFCatalogMatching)*
               
## Installation
The latest version of `SkyLink` can be installed as follows:
```
pip install git+https://github.com/enourbakhsh/skylink
```
I strongly recommend doing this in a conda virtual environment.

## Common Installation Issues
This package takes advantage of `f-strings` which were introduced with Python 3.6. In older python versions, an f-string will result in a syntax error.

Make sure you have an updated version of `setuptools` before installing `SkyLink`. A simple `pip install setuptools -U` does it for you.

For most users, all the dependencies will install automatically. However, some might encounter issues with the `networkit` package. If you are one of those users, try `conda install networkit` and if that fails follow the instructions [here](https://github.com/networkit/networkit).
`networkit` requires [CMake](https://cmake.org/install/) 3.5 or higher. Make sure you check this by running the command `cmake --version` in your terminal. 

In case `pip` gives you a `gcc` error while installing `psutil` in a `conda` environment, try `conda install psutil` before installing `SkyLink` ([source](https://github.com/ray-project/ray/issues/1340)).

## Testing and maintainability
In the root directory, you can run the unit tests available in the `tests` directory using the `pytest` command (assuming you have `pytest` installed):
```bash
pytest .
```

While in the root directory, if you only want to check the code base against coding style (PEP8), programming errors and to check cyclomatic complexity, you can run this command (assuming you have `flake8` installed):
```bash
flake8 .
```

## Citing SkyLink
You can cite `SkyLink` using the following BibTex reference format:

```bibtex
@software{Nourbakhsh2020,
title={A parallel Python code for efficiently matching sky catalogs using KDTrees and graphs},
url={https://github.com/enourbakhsh/skylink},
author={Erfan Nourbakhsh},
version={1.0.0},
year={2020}}
```
