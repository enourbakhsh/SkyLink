"""Skylink
Code for efficiently matching sky catalogs using KDTrees and graphs. This
includes internal matching via friends-of-friends algorithm.

Notes
-----
(1) This package uses the `FoFCatalogMatching` package as a benchmark to verify
the results. That's why I adopted some codes and also the style of the outputs
from the aforementioned package.

(2) `linking_length` as a dictionary (N-way matching) has not yet been tested!
"""

__all__ = ["match"]

import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from .fof import fastmatch
from busypal import BusyPal
import pandas as pd
import pickle
import os
import sys
import subprocess
import inspect
import time
import colored as cl
import datetime

fof_path = inspect.getfile(fastmatch)

# MPI (experimental)
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()


def _run_command(cmd, points, points_path, group_ids_path):
    """Runs the comand to execute MPI

    Parameters
    ----------
    cmd : str
        MPI command
    points : `~astropy.SkyCoord`
        The stacked catalog obtained from all catalogs
    points_path : str
        The path to strore points in
    group_ids_path : str
        The path to store the grouped results in

    Returns
    -------
    group_id: int array
        An array of group ids

    Raises
    -------
    RuntimeError
        Raised if the process exited with code != 0
    """

    # Remove the old results just in case.
    if os.path.exists(group_ids_path):
        os.remove(group_ids_path)
    with open(points_path, "wb") as h:
        pickle.dump(points, h)
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    while True:
        line = process.stdout.readline()
        if "%|" in line:  # tqdm line
            print(f"\r{line.rstrip()}", end="", flush=True)
        else:
            print(f"{line.rstrip()}")
        if not line:  # EOF
            returncode = process.poll()
            if returncode is not None:
                break
        sys.stdout.flush()
        time.sleep(0.02)  # cmd closed the stdout, but not exited yet

    return_code = process.poll()

    if return_code != 0:
        raise RuntimeError(
            f"Something went wrong in '{fof_path}' with the return code {return_code}"
        )

    if os.path.exists(points_path):
        os.remove(points_path)
    with open(group_ids_path, "rb") as h:
        group_id = pickle.load(h)
    os.remove(group_ids_path)
    return group_id


def _check_max_count(count):
    """Checks if the input value is valid (either None or a positive >= 1
    number.

    Parameters
    ----------
    count : int
        Input integer.

    Returns
    -------
    count: int
        Output integer. Returns the input integer if valid.

    Raises
    -------
    ValueError
        If `count` is not None or a positive number.

    """
    if count is not None:
        count = int(count)
        if count < 1:
            raise ValueError("`count` must be None or a positive integer.")
        return count


def match(
    catalog_dict,
    linking_lengths=None,
    ra_label="ra",
    dec_label="dec",
    ra_unit="deg",
    dec_unit="deg",
    catalog_len_getter=len,
    mpi=False,
    mpi_path="mpirun",
    graph_lib="igraph",
    num_threads=None,
    nprocs=2,
    overlap=1.0,
    cache_root=os.getcwd(),
    sort=True,
    return_pandas=False,
    storekdtree=True,
    use_linked_mask=True,
    verbose=1,
    show_progress=True,
    silent=False,
    **tqdm_kwargs,
):
    """Match astronomical sky catalogs. Made to be consistent with the
    `FoFCatalogMatching` package.

    Parameters
    ----------
    catalog_dict : dict
        Catalogs to match.
        In the format of {'cat_a': catalog_table_a, 'cat_b': catalog_table_b}
    linking_lengths : dict or float
        FoF linking length. Assuming the unit of arcsecond.
        Can specify multiple values with the maximal allowed numbers in each
        group.
        Use `None` to mean to constraint.
        Example: {5.0: 5, 4.0: 5, 3.0: 4, 1.0: None}
    ra_label : str, optional, default: 'ra'
        The label for the ra column
    dec_label : str, optional, default: 'dec'
        The label for the dec column
    ra_unit : str or astropy.units.Unit, optional, default: 'deg'
        The unit of ra coordinates.
    dec_unit : str or astropy.units.Unit, optional, default: 'deg'
        The unit of dec coordinates.
    catalog_len_getter : callable, optional, default: len
        A callable that returns the length of the catalog.
    mpi : bool, optional, default: False
        Whether to use MPI for parallelization.
    mpi_path : str, default: 'mpirun'
        Path to the MPI executable
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    num_threads : int, optional, default: None
        Number of OpenMP threads (only applies to the `networkit` library)
    nprocs : int, optional, default: 2
        Number of processors to use
    overlap : float, optional, default: 1.0
        The amount of overlaps for the mosaics in the unit of the FoF linking
        length.
    cache_root : str, optional, default: os.getcwd()
        Root path to cache the stacked catalog
    sort : bool, optional, default: True
        Whether to sort the catalog at the end in terms of the group id and row
        index.
    return_pandas : bool, optional, default: False
        Return the results as a pandas dataframe, otherwise an astropy table
    storekdtree : bool or str, optional, default: True
        An astropy parameter. If a string, will store the KD-Tree used in the
        search with the name ``storekdtree`` in ``coords2.cache``.
        This speeds up subsequent calls to this function. If False, the
        KD-Trees are not saved.
    use_linked_mask : bool, optional, default: True
        If True, it generates a mask to be applied to the arrays from different
        parallel processes before stitching them together. This reduces the
        time to create a graph in ``stitch_group_ids()`` but might have some
        minimal amount of overhead while making the mask through
        ``get_mosaics()`` and ``get_group_ids()``.
    verbose : bool or int, optional, default: True
        If True or 1, it produces lots of logging output.
    show_progress : bool or int, optional, default: True
        Show ``tqdm`` progressbars.
    silent : bool or int, optional, default: False
        Run the code silently by suppressing the logging output and the
        progressbar.
    **tqdm_kwargs : dict
        Additional keyword arguments passed to ``tqdm``.

    Returns
    -------
    stacked_catalog: `astropy.table.Table`
        A table containing group id and row id for each catalog.
    """

    t0 = datetime.datetime.now()

    if verbose:
        if nprocs > 1:
            print(
                cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
                + f" Running {nprocs} parallel jobs"
            )
        elif nprocs == 1:
            print(
                cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
                + " Running without parallelization"
            )
        else:
            raise ValueError("illegal `nproc`")

    if not show_progress:
        skip_busypal = 1
        # disable_tqdm = True
    else:
        skip_busypal = 0
        # disable_tqdm = False

    if silent:
        verbose = 0
        skip_busypal = 2
        # disable_tqdm = True

    if mpi:
        raise NotImplementedError(
            "MPI is an experimental feature that has not yet been tested."
        )
        if not cache_root == "" and not cache_root.endswith("/"):
            cache_root += "/"
        points_path = cache_root + "points.cache"
        group_ids_path = cache_root + "group_ids.cache"

    if isinstance(linking_lengths, dict):
        linking_lengths = [
            (float(k), _check_max_count(linking_lengths[k]))
            for k in sorted(linking_lengths, key=float, reverse=True)
        ]
    else:
        linking_lengths = [(float(linking_lengths), None)]

    xstacked_catalog = []
    for catalog_key, catalog in catalog_dict.items():
        if catalog is None:
            continue

        n_rows = catalog_len_getter(catalog)
        xstacked_catalog.append(
            Table(
                {
                    "ra": catalog[ra_label],
                    "dec": catalog[dec_label],
                    "row_index": np.arange(n_rows),
                    "catalog_key": np.repeat(catalog_key, n_rows),
                }
            )
        )

    if not xstacked_catalog:
        raise ValueError("No catalogs to merge!")

    stacked_catalog = vstack(xstacked_catalog, "exact", "error")
    points = SkyCoord(
        stacked_catalog["ra"], stacked_catalog["dec"], unit=(ra_unit, dec_unit)
    )  # .cartesian.xyz.value.T

    # TODO: faster non-internal match i.e. when you don't need fof
    coords1 = None
    coords2 = None

    del stacked_catalog["ra"], stacked_catalog["dec"]

    group_id = regroup_mask = group_id_shift = None

    for linking_length_arcsec, max_count in linking_lengths:

        if group_id is None:
            if mpi:
                cmd = f"{mpi_path} -n {nprocs} {sys.executable} {fof_path} --points_path={points_path} \
                        --linking_length={linking_length_arcsec} --group_ids_path={group_ids_path} \
                        --tqdm_kwargs={tqdm_kwargs}"
                print(f"Running the command: {cmd}")
                group_id = _run_command(cmd, points, points_path, group_ids_path)
            else:
                group_id = fastmatch(
                    coords=points,
                    coords1=coords1,
                    coords2=coords2,
                    linking_length=linking_length_arcsec,
                    reassign_group_indices=False,
                    graph_lib=graph_lib,
                    num_threads=num_threads,
                    storekdtree=storekdtree,
                    use_linked_mask=use_linked_mask,
                    njobs=nprocs,
                    verbose=verbose,
                    show_progress=show_progress,
                    silent=silent,
                    **tqdm_kwargs,
                )
        else:
            if mpi:
                # if `reassign_group_indices=False` by default in fof.py's
                # argparse, you can set the flag --reassign_group_indices to
                # make it True.
                cmd = [
                    f"{mpi_path} -n {nprocs}",
                    sys.executable,
                    fof_path,
                    f"--points_path={points_path}",
                    f"--linking_length={linking_length_arcsec}",
                    f"--group_ids_path={group_ids_path}",
                    f"--tqdm_kwargs={tqdm_kwargs}",
                ]
                group_id = _run_command(
                    cmd, points[regroup_mask], points_path, group_ids_path
                )
            else:
                group_id[regroup_mask] = fastmatch(
                    points=points[regroup_mask],
                    linking_length=linking_length_arcsec,
                    reassign_group_indices=False,
                )

            group_id[regroup_mask] += group_id_shift

        if max_count is None:
            _, group_id = np.unique(group_id, return_inverse=True)
            break

        with BusyPal(
            "Reassigning group ids with consecutive numbers",
            skip=skip_busypal,
            verbose=verbose,
        ):
            _, group_id, counts = np.unique(
                group_id, return_inverse=True, return_counts=True
            )
        group_id_shift = group_id.max() + 1
        regroup_mask = counts[group_id] > max_count
        del counts

        if not regroup_mask.any():
            break

    group_id = pd.factorize(group_id)[0]  # very fast!
    stacked_catalog["group_id"] = group_id

    if sort:
        with BusyPal("Sorting", skip=skip_busypal, verbose=verbose):
            stacked_catalog = stacked_catalog.group_by(["group_id", "row_index"])

    if return_pandas:
        if verbose:
            elapsed_seconds = round((datetime.datetime.now() - t0).seconds)
            print(
                cl.stylize("✔ Success!", cl.fg("green") + cl.attr("bold"))
                + f" Took {str(datetime.timedelta(seconds=elapsed_seconds))} hms."
            )
        return stacked_catalog.to_pandas()
    else:
        if verbose:
            elapsed_seconds = round((datetime.datetime.now() - t0).seconds)
            print(
                cl.stylize(
                    f"✔ Success! Took {str(datetime.timedelta(seconds=elapsed_seconds))} to execute.",
                    cl.attr("bold") + cl.fg("green"),
                )
            )
        return stacked_catalog
