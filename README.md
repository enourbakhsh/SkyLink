# SkyLink
Code for efficiently matching sky catalogs using KDTrees and graphs. This includes internal matching via friends-of-friends algorithm. Even in its serial mode, `SkyLink` performs faster than many other approaches I came across (e.g. the `FoFCatalogMatching` package hosted [here](https://github.com/yymao/FoFCatalogMatching)).
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
`SkyLink` can be installed as follows:
```
pip install git+https://github.com/enourbakhsh/skylink
```

## Common Issues
For most users, all the dependencies will install automatically. However, some might encounter issues with the `networkit` package. If you are one of those users, please follow the instructions here: https://github.com/networkit/networkit.

`networkit` requires CMake 3.5 or higher. Make sure you check this by running the command `cmake --version` in your terminal. 

## Citing
You can cite `SkyLink` using the following BibTex reference format:

```
@misc{Nourbakhsh2020,
title={A parallel Python code for efficiently matching sky catalogs using KDTrees and graphs},
url={https://github.com/enourbakhsh/skylink},
author={Erfan Nourbakhsh},
year={2020}}
```
