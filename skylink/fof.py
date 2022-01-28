import sys
import numpy as np
from .astropy_search.matching import search_around_sky
from .graph import GraphDataStructure
from astropy.coordinates import Angle
from itertools import chain
import datetime
from functools import partial
import multiprocessing
from busypal import busy, BusyPal, session
import pandas as pd
import colored as cl

# TODO:  Implement MPI - almost there in `skylink.py`!
# TODO:  Make a function to stitch overlapping patches with pre-calculated
#        `group_ids` in general.
# FIXME: Progressbars and busy indicators (in particular the ones that use
#        decorators) still show up with `silent=True`
# FIXME: Try with 100 or smaller - sometimes (rarely) it hangs eternally!

# Complex Network Analysis: The Need for Speed (Benchmark Paper)
# http://m3nets.de/publications/CCC2016d.pdf

__all__ = ["fastmatch"]


# Uses https://stackoverflow.com/questions/62759478/fastest-way-to-update-a-list-of-integers-based-on-the-duplicates-in-another-list  # noqa
def update_labels(items, labels):
    """Update a list of integers of the same length based on the duplicates in
    another list of integers. If an item is duplicated in the list of items,
    that means they are labeled using different integers in the list of labels.
    This assigns the same int/label (the label of the first occurrence) to
    all of the items that are labeled with those integers. Note that this can
    be more than just the duplicates we first found in the list of items.

    Parameters
    ----------
    items : int list or int array
        The list of items.
    labels : int list or int array
        The list of labels.

    Returns
    -------
    labels: int list or int array
        Updated labels.

    Example
    -------
    >>> items = np.array([7, 2, 0, 6, 0, 4, 1, 5, 2, 0])
    >>> labels = np.array([1, 0, 3, 4, 2, 1, 6, 6, 5, 4])
    >>> update_labels(items, labels)
    [1 0 3 4 3 1 6 6 0 3]
    """

    i_dict, l_dict, ranks = {}, {}, {}

    for i in range(len(items)):
        label = i_dict.setdefault(items[i], labels[i])
        if labels[i] not in ranks:
            ranks[labels[i]] = i

        if label != labels[i]:
            label1 = label
            label2 = labels[i]
            while label1 is not None and label2 is not None:
                if ranks[label1] > ranks[label2]:
                    tmp = l_dict.get(label1)
                    l_dict[label1] = label2
                    label1 = tmp
                elif ranks[label1] < ranks[label2]:
                    tmp = l_dict.get(label2)
                    l_dict[label2] = label1
                    label2 = tmp
                else:
                    break

            labels[i] = label

    for i in range(len(labels)):
        val = 0
        label = labels[i]
        while val != -1:
            val = l_dict.get(label, -1)
            if val != -1:
                label = val
        if label != labels[i]:
            labels[i] = label

    return labels


def stitch_group_ids(
    items, labels, graph_lib="networkit", verbose=True, num_threads=None
):
    """Stitch and unify group ids found from parallel processes.

    Parameters
    ----------
    items : int list or int array
        The list of items.
    labels : int list or int array
        The list of labels.
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    verbose : bool or int, optional, default: True
        If True or 1, it produces lots of logging output.
    num_threads : int, optional, default: None
        Number of OpenMP threads (only applies to `networkit`)

    Returns
    -------
    new_labels : int array
        The unified array of group ids from all processes.
    """

    labels_max = labels.max()  # max() was slower!

    # We mark our items so that we know what to remove from clustered integers
    # later on.
    items = items + labels_max + 1
    edges = zip(items, labels)

    nnodes = (
        items.max() + labels_max + 1 if graph_lib == "networkit" else None
    )  # not needed otherwise
    gds = GraphDataStructure(
        graph_lib, num_threads=num_threads
    )  # num_threads is for `networkit` only
    graph = gds.build_graph_from_edges(edges, verbose=verbose, nnodes=nnodes)
    del edges
    cc = find_clusters(graph, graph_lib=graph_lib, verbose=False)

    a = list(map(list, (x for x in cc if len(x) > 2)))
    del cc
    a = [
        [w for w in x if w <= labels_max] for x in a
    ]  # slowest [[w for w in x if not isinstance(w, str)] for x in a]

    b = [[x[0]] * (len(x) - 1) for x in a]
    for x in a:
        del x[0]

    # This is orders of magnitude faster than np.concatenate(a) in this case
    # with the lists.
    a = list(
        chain(*a)
    )
    b = list(chain(*b))

    mapping = np.arange(0, labels_max + 1)
    mapping[a] = b
    new_labels = mapping[labels]

    return new_labels


# Built on https://python.hotexamples.com/examples/astropy.coordinates/SkyCoord/to_pixel/python-skycoord-to_pixel-method-examples.html # noqa
def radec2xy(
    coords, ra_unit="deg", dec_unit="deg", wcs=None, mode="all", pixel_scale=1
):
    """Project a catalog onto the image plane. The default is the tangent image
    plane, but any WCS transformation can be used.

    Parameters
    ----------
    coords : astropy SkyCoord, Quantity, or Angle
        The coordinate list. If a ``Quantity`` or ``Angle``, ``cat`` must be
        a 2xN array as (lat, long).
    ra_unit : str or astropy.units.Unit, optional, default: 'deg'
        The unit of ra coordinates.
    dec_unit : str or astropy.units.Unit, optional, default: 'deg'
        The unit of dec coordinates.
    wcs : `~astropy.wcs.WCS`, optional, default: None
        The world coordinate system object for transformation.  The
        default assumes the tangent plane centered on the latitude
        and longitude of the catalog.
    mode : string, optional, default: 'all'
        The projection mode for `SkyCoord` objects: 'wcs' or 'all'.  See
        `~astropy.SkyCoord.to_pixel`.
    pixel_scale : int, optional, default: 1
        The pixl scale in pixels per square arcsecond.

    Returns
    -------
    x, y : `numpy.ndarray`
        The pixel coordinates.

    Raises
    ------
    ValueError
        If there are non-finite values in the projected coordinates or the
        cellestial coordinates are not gnomonic-projectable onto a plane.
    """

    # See https://github.com/astropy/astropy/issues/2847
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    if not isinstance(coords, SkyCoord):
        assert len(coords) == 2
        coords = SkyCoord(ra=coords[0], dec=coords[1], unit=(ra_unit, dec_unit))

    # Create the projection without having to deal with FITS files.
    if wcs is None:
        wcs = WCS(naxis=2)
        # pixel coordinate of the reference point
        # --> the projection is centered at 0, 0
        wcs.wcs.crpix = (0, 0)
        ra_tan = (coords.ra.max() - coords.ra.min()) / 2.0 + coords.ra.min()
        dec_tan = (coords.dec.max() - coords.dec.min()) / 2.0 + coords.dec.min()
        # Coordinate value at the reference point.
        wcs.wcs.crval = (
            ra_tan.value,
            dec_tan.value,
        )
        # Axis type
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # Coordinate increment for pixel_scale=1
        # The pixel scale is 1"/pixel.
        # This outputs values in arcsecond offset.
        wcs.wcs.cdelt = [pixel_scale / 3600.0, pixel_scale / 3600.0]

    # x, y = wcs.wcs_world2pix(coords.ra,coords.dec,0)
    x, y = coords.to_pixel(wcs, mode=mode)

    if (~np.isfinite(x)).sum() != 0:
        raise ValueError(
            "We encountered one or more non-finite values in the projected coordinates."
        )

    if (~np.isfinite(x)).sum() == len(x):
        raise ValueError(
            "The projection led to all non-finite pixel coordinates, probably because it did not converge."
            "Make sure your ra and dec ranges are valid (0<ra<360 and -45<dec<45 in degrees) and"
            "gnomonic-projectable onto a plane."
            "Also see https://docs.astropy.org/en/stable/_modules/astropy/wcs/wcs.html"
        )

    return x, y


# `tqdm` automatically switches to the text-based progress bar if not running
# in Jupyter

try:  # https://github.com/tqdm/tqdm/issues/506
    ipy_str = str(type(get_ipython()))  # noqa
    if "zmqshell" in ipy_str:  # jupyter
        from tqdm.notebook import tqdm
        # TODO: the following needs to be tested first
        # # https://github.com/bstriner/keras-tqdm/issues/21
        # # this removes the vertical whitespace that remains when tqdm
        # # progressbar disappears
        # from IPython.core.display import HTML
        # HTML("""
        # <style>
        # .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
        #   padding: 0;
        #   border: 0;
        # }
        # </style>
        # """)
    elif "terminal" in ipy_str:  # ipython
        from tqdm import tqdm
    else:
        pass
except BaseException:  # terminal
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:

        def tqdm(iterable, **kwargs):
            """ dummy tqdm """
            return iterable


def get_group_ids(
    coords=None,
    coords1=None,
    coords2=None,
    coords_idxshift=None,
    coords1_idxshift=None,
    coords2_idxshift=None,
    overidx=None,
    overidx1=None,
    overidx2=None,
    storekdtree=None,
    job_num=0,
    linking_length=None,
    graph_lib="igraph",
    num_threads=None,
    num_objects=None,
    verbose=1,
    show_progress=True,
    silent=False,
    tqdm_kwargs={},
):
    """Returns unique ids for objects belonging to each group.

    Parameters
    ----------
    coords : `~astropy.SkyCoord`
        The stacked catalog obtained from all catalogs (not optional in this
        version).
        TODO: make it optional so that the user can pass coords1 and coords2
        instead.
    coords1 : `~astropy.SkyCoord`, optional, default: None
        The first catalog.
    coords2 : `~astropy.SkyCoord`, optional, default: None
        The second catalog.
    coords_idxshift : int
        Make the group ids of each chunk very far from other chunks so that
        there is absolutely no chance of a conflict. We can reassign sorted
        consecutive indices to the objects after we combine them at the end.
    coords1_idxshift : int
        Same as `coords_idxshift` but for the first catalog in the cross
        catalog mode (TODO).
    coords2_idxshift : int
        Same as `coords_idxshift` but for the second catalog in the cross
        catalog mode (TODO).
    overidx :
        Indices for duplicate groups to the input catalog.
        The reason we encounter some duplicates is because our little mosaics
        are designed to overlap.
    overidx1 :
        Same as `overidx` but for the first catalog in the cross
        catalog mode (TODO).
    overidx2 :
        Same as `overidx` but for the second catalog in the cross
        catalog mode (TODO).
    storekdtree : bool or str, optional, default: None
        An astropy parameter. If a string, will store the KD-Tree used in the
        search with the name ``storekdtree`` in ``coords2.cache``.
        This speeds up subsequent calls to this function. If False, the
        KD-Trees are not saved.
    job_num : int, default: 0
        Job number.
    linking_length : float
        FoF linking length. Assuming the unit of arcsecond.
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    num_threads : int, optional, default: None
        Number of OpenMP threads (only applies to the `networkit` library)
    num_objects : int, default: None
        Number of object in the input catalog.
    verbose : bool or int, optional, default: 1
        If True or 1, it produces lots of logging output.
    show_progress : bool or int, optional, default: True
        Show ``tqdm`` progressbars.
    silent : bool or int, optional, default: False
        Run the code silently by suppressing the logging output and the
        progressbar.
    tqdm_kwargs : dict, default: {}
        A dictionary containing additional keyword arguments used for ``tqdm``.

    Returns
    -------
    group_ids : int array
        Unique ids for objects belonging to each group.

    Notes
    -----
    (1) For the future implementation of cross catalog mode. coord1 should not
        be partial but coord2 can be partial.

    (2) One good thing about `networkx`: it does not make unnecessary isolated
        nodes which means faster performance at least in the serial mode.
        The other two graph libraries do make isolated nodes. So networkx is
        better for when we expect many isolated coordinates in a huge dataset.

    (3) Despite (2), `networkx` is written purely in python, the other two are
        written in C/C++. `networkit` has the added benefit of OpenMP.

    (4) `igraph` and `networkx` add the edges at onece but `networkit` needs to
        use a loop to add them one by one. In fact, `networkit`'s C and R
        version has an `addlist()` that does it at once but I couldn't find the
        same functionality in their python version.

    (5) Check the new versions of these libraries to see if they added more
        capabilities.
    """

    if job_num == -1:
        job_num = 0
        parallel = False
    else:
        parallel = True

    tqdm_kwargs["position"] = job_num

    # if job_num!=0:
    #     verbose=False
    #     silent=True

    if job_num != 0:
        silent = True

    skip_busypal = -1 if show_progress else 1

    if silent:
        verbose = 0
        skip_busypal = 2
        disable_tqdm = True
    else:
        disable_tqdm = False

    idx1, idx2 = search_around_sky(
        coords=coords,
        coords1=coords1,
        coords2=coords2,
        seplimit=Angle(f"{linking_length}s"),
        storekdtree=storekdtree,
        verbose=verbose,
        show_progress=show_progress,
        silent=silent,
        tqdm_kwargs=tqdm_kwargs,
    )

    if verbose:
        pass

    # Delete duplicates and 1:1 singles. (It is important that a<b is enforced
    # for `networkit` nnodes finder to work)

    graph_edges = set((a, b) if a < b else (b, a) for a, b in zip(idx1, idx2) if a != b)

    num_objects_chunk = (
        len(coords) if coords is not None else len(coords1) + len(coords2)
    )

    with BusyPal(
        "Building the representative graph/network",
        style={"id": 6, "color": "sandy_brown"},
        fmt="{spinner} {message}",
        skip=skip_busypal,
    ):
        gds = GraphDataStructure(
            graph_lib, num_threads=num_threads
        )  # Threads are for `networkit` only.
        final_graph = gds.build_graph_from_edges(
            graph_edges, verbose=False, nnodes=num_objects_chunk
        )
        clusters = find_clusters(final_graph, graph_lib=graph_lib, verbose=False)

    nclusters = len(clusters)

    # Make ids very far from each other in the nmber space (absolutely no
    # chance of a conflict).
    starting_id = (job_num + 2) * num_objects + coords_idxshift
    group_ids = np.arange(starting_id, starting_id + num_objects_chunk)
    linked_mask = np.zeros(num_objects_chunk, dtype=bool)

    del tqdm_kwargs["desc"]
    if overidx is not None:
        # This makes the lookups very fast.
        overidx = set(overidx)
        for idx, cluster in enumerate(
            tqdm(
                clusters,
                total=nclusters,
                desc="Finding connected components of the"
                + "graphs and shared components"
                if parallel
                else "graph",
                disable=disable_tqdm,
                **tqdm_kwargs,
            )
        ):
            # If any of the connected components has a foot on the overlap
            # region, that whole group should be involved in the stitching
            # process with another mosaic later.
            if any(gidx in overidx for gidx in cluster):
                linked_mask[cluster] = True
            group_ids[cluster] = idx + coords_idxshift
        del clusters
        if verbose:
            print(
                "\r\r"
                + cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
                + " Assigned group ids for each chunk by using connected components of the graphs"
                if parallel
                else "by using connected components of the graph"
            )
        return group_ids, linked_mask
    else:
        # It might still be parallel but it is not using the linked mask.
        for idx, cluster in enumerate(
            tqdm(
                clusters,
                total=nclusters,
                desc="Finding connected components of the" + "graphs"
                if parallel
                else "graph",
                disable=disable_tqdm,
                **tqdm_kwargs,
            )
        ):
            group_ids[cluster] = idx + coords_idxshift
        del clusters
        if verbose:
            print(
                "\r\r"
                + cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
                + " Assigned group ids for each chunk"
                if parallel
                else " Assigned group ids"
            )
        return group_ids


def find_clusters(graphs, graph_lib="igraph", verbose=True):
    """Find clusters of close pairs and multiples in a graph using a connected
    compponent analyis.

    Parameters
    ----------
    graphs : `graph` object of `graph_lib`
        The input graph. It can be a list/tuple of graphs or just a single
        graph.
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    verbose : bool or int, optional, default: True
        If True or 1, it produces lots of logging output.

    Returns
    -------
    clusters : `cluster` object of `graph_lib`
        Clusters found using a connected component analysis.
    """

    t0 = datetime.datetime.now()
    gds = GraphDataStructure(graph_lib)
    # FIXME: address num_threads=? for `networkit` only

    # Merge and cluster (it does the merging internally if you pass a list or
    # tuple of graphs)
    clusters = gds.cluster(graphs, verbose=verbose)
    if verbose:
        elapsed_seconds = round((datetime.datetime.now()-t0).seconds)
        print(
            f"clustering done in {str(datetime.timedelta(seconds=elapsed_seconds))} hms."
        )

    # Remove isolated points.
    # t0 = datetime.datetime.now()
    # clusters = list(filter(lambda x: len(x)>1, clusters))
    # clusters = [sl for sl in clusters if len(sl)>1] # not as elegant
    # print(f'removing isolated clusters for {graph_lib} generated cluster done
    # in {str(datetime.timedelta(
    # seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')

    return clusters


# TODO: make the following parallel! We might not get that much boost from it.
@busy(
    "Mosaicking data",
    style={"id": 6, "color": "sandy_brown"},
    fmt="{spinner} {message}",
)
def get_mosaic_sets(
    coords=None,
    coords1=None,
    coords2=None,
    linking_length=None,
    wcs=None,
    mode="all",
    nside_mosaics=None,
    njobs=None,
    overlap=1.0,
    use_linked_mask=False,
):
    """Identify mosaic sets for the input catalog.

    Parameters
    ----------
    coords : `~astropy.SkyCoord`
        The stacked catalog obtained from all catalogs (not optional in this
        version).
        TODO: make it optional so that the user can pass coords1 and coords2
        instead.
    coords1 : `~astropy.SkyCoord`, optional, default: None
        The first catalog.
    coords2 : `~astropy.SkyCoord`, optional, default: None
        The second catalog.
    linking_lengths : dict or float
        FoF linking length. Assuming the unit of arcsecond.
        Can specify multiple values with the maximal allowed numbers in each
        group.
        Use `None` to mean to constraint.
        Example: {5.0: 5, 4.0: 5, 3.0: 4, 1.0: None}
    wcs : astropy.wcs.WCS, optional, default: None
        The world coordinate system object for transformation. The
        default assumes the tangent plane centered on the latitude
        and longitude of the catalog.
    mode : string, optional, default; 'all'
        The projection mode for `SkyCoord` objects: 'wcs' or 'all'.  See
        `SkyCoord.to_pixel`.
    nside_mosaics : int, optional, default: `int(2 * np.sqrt(njobs))`
        Resolution parameter for the mosaics. The number of mosaics would be
        roughly nside_mosaics**2.
    njobs : int, optional, default: 1
        Number of jobs to run in parallel.
    overlap : float, optional, default: 1.0
        The amount of overlaps for the mosaics in the unit of the FoF linking
        length.
    use_linked_mask : bool, optional, default: False
        If True, it generates a mask to be applied to the arrays from different
        parallel processes before stitching them together. This reduces the
        time to create a graph in ``stitch_group_ids()`` but might have some
        minimal amount of overhead while making the mask through
        ``get_mosaics()`` and ``get_group_ids()``.

    Returns
    -------
    coords_chunks : float array
        Chunked up version of the input catalog.
    refidx_chunks : int array
        Chunked up version of the reference indecies from the input catalog.
    overidx_chunks : int array
        Gives indices for duplicate groups to the chunk array not the full
        reference catalog.
        The reason we encounter some duplicates is because our little mosaics
        are designed to overlap.

    Raises
    ------
    NotImplementedError
        If coords is not supplied. You can't use coords1 and coords2 yet.
    """

    # x and y are the projection of ra and dec onto the image plane.
    # We temporarilly made them for splitting purposes.
    x, y = radec2xy(coords, wcs=wcs, mode=mode)

    if nside_mosaics is None:
        # `nside_mosaics` is a function of `njobs`
        # Each job takes care of multiple mosaics.
        nside_mosaics = int(2 * np.sqrt(njobs))

    H, xedges, yedges = np.histogram2d(x, y, bins=nside_mosaics)
    idx_filled_mosaics = np.where(H > 0)
    num_filled_mosaics = len(idx_filled_mosaics[0])
    xbounds, ybounds, refidx_inside, refidx_overlap = [], [], [], []

    for idx_x, idx_y in zip(*idx_filled_mosaics):
        xbounds.append([xedges[idx_x], xedges[idx_x + 1]])
        ybounds.append([yedges[idx_y], yedges[idx_y + 1]])

    for xbound, ybound in zip(xbounds, ybounds):
        # Want to create overlapping mosaics?
        # If yes, technically 0.5*l (overlap=1.0) leads to one linking length
        # overlap, but the user can choose to be more conservative.
        x0 = xbound[0] - linking_length * float(overlap) / 2.0
        x1 = xbound[1] + linking_length * float(overlap) / 2.0
        y0 = ybound[0] - linking_length * float(overlap) / 2.0
        y1 = ybound[1] + linking_length * float(overlap) / 2.0
        cx0 = x >= x0
        cx1 = (
            x <= x1 if x1 == xedges[-1] else x < x1
        )  # x < x1 is enough if overlapping_mosaics = False
        cy0 = y >= y0
        cy1 = (
            y <= y1 if y1 == yedges[-1] else y < y1
        )  # y < y1 is enough if overlapping_mosaics = False
        refidx_inside.append(np.where(cx0 & cx1 & cy0 & cy1)[0])

        # idx_inside = np.where(cx0 & cx1 & cy0 & cy1) # integers not bools
        # coords_mosaics.append(coords[idx_inside])

        if use_linked_mask:
            # Find the internal bounds this mosaic makes with other overlapping
            # mosaics. This information is used for stitching the chunks of
            # `group_id` later after concatenating results from different
            # processes.
            x0_p = xbound[0] + linking_length * float(overlap) / 2.0
            x1_p = xbound[1] - linking_length * float(overlap) / 2.0
            y0_p = ybound[0] + linking_length * float(overlap) / 2.0
            y1_p = ybound[1] - linking_length * float(overlap) / 2.0
            cx0_p = x <= x0_p  # (x0<=x) & (x<=x0_p)
            cx1_p = x1_p <= x  # ...
            cy0_p = y <= y0_p  # (y0<=y) & (y<=y0_p)
            cy1_p = y1_p <= y  # ...
            refidx_overlap.append(
                np.where((cx0 & cx1 & cy0 & cy1) & (cx0_p | cx1_p | cy0_p | cy1_p))[0]
            )

    idx_mosaics_for_chunks = np.array_split(range(num_filled_mosaics), njobs)

    # Chunks can consist of one or more mosaics which are assigned to each job
    coords_chunks, refidx_chunks = [], []
    overidx_chunks = [] if use_linked_mask else [None] * njobs
    for idx_mosaics in idx_mosaics_for_chunks:
        refidx_inside_unified = np.array([], dtype=np.int64)
        if use_linked_mask:
            refidx_overlap_unified = np.array([], dtype=np.int64)
        for m in idx_mosaics:
            refidx_inside_unified = np.append(refidx_inside_unified, refidx_inside[m])
            if use_linked_mask:
                refidx_overlap_unified = np.append(
                    refidx_overlap_unified, refidx_overlap[m]
                )
        # There might be some duplicates since our little mosaics have
        # some overlaps.
        refidx_inside_unified = np.unique(refidx_inside_unified)
        coords_chunks.append(coords[refidx_inside_unified])
        # Indices to the main/reference catalog.
        refidx_chunks.append(refidx_inside_unified)
        if use_linked_mask:
            # There might be some duplicates due to overlapping mosaics.
            refidx_overlap_unified = np.unique(refidx_overlap_unified)
            idx_overlap_unified = np.where(
                np.isin(refidx_inside_unified, refidx_overlap_unified)
            )[
                0
            ]  # Gives indices to the chunk array (not the reference catalog).
            overidx_chunks.append(idx_overlap_unified)

    if coords is not None:
        return (
            coords_chunks,
            refidx_chunks,
            overidx_chunks,
        )  # `refidx` indices to the main catalog.
    else:
        raise NotImplementedError('Runs are not defined for when coords is None')  # TODO
        # return (
        #     coords1_chunks,
        #     coords2_chunks,
        #     refidx_chunks,
        #     overidx_chunks,
        # )


def fastmatch(
    coords=None,
    coords1=None,
    coords2=None,
    linking_length=None,
    reassign_group_indices=True,
    njobs=1,
    overlap=1.0,
    graph_lib="igraph",
    num_threads=None,
    storekdtree=True,
    use_linked_mask=False,
    verbose=1,
    show_progress=True,
    silent=False,
    **tqdm_kwargs,
):
    """Performs an efficient catalog matching using KDTrees and graphs.

    Parameters
    ----------
    coords : `~astropy.SkyCoord`
        The stacked catalog obtained from all catalogs (not optional in this
        version).
        TODO: make it optional so that the user can pass coords1 and coords2
        instead.
    coords1 : `~astropy.SkyCoord`, optional, default: None
        The first catalog.
    coords2 : `~astropy.SkyCoord`, optional, default: None
        The second catalog.
    linking_lengths : dict or float
        FoF linking length. Assuming the unit of arcsecond.
        Can specify multiple values with the maximal allowed numbers in each
        group.
        Use `None` to mean to constraint.
        Example: {5.0: 5, 4.0: 5, 3.0: 4, 1.0: None}
    reassign_group_indices : bool, optional, default: True
        Assign sorted consecutive group ids to the objects at the end.
        It will be a continuous sequence from 0 to n-1 with n being the number
        of groups.
    njobs : int, optional, default: 1
        Number of jobs to run in parallel.
    overlap : float, optional, default: 1.0
        The amount of overlaps for the mosaics in the unit of the FoF linking
        length.
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    num_threads : int, optional, default: None
        Number of OpenMP threads (only applies to the `networkit` library)
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
    group_id : int array
        An array of assigned group ids with the same order as the input
        catalog.
        TODO: If `coords1` and `coords2` is supplied instead of the stacked
        `coords` object, the order would be `coords1` and then `coords2`.
    """

    # Define aliases for graph library names.
    if graph_lib == "nk":
        graph_lib = "networkit"
    elif graph_lib == "nx":
        graph_lib = "networkx"
    elif graph_lib == "ig":
        graph_lib = "igraph"

    if use_linked_mask and graph_lib == "networkx":
        raise ValueError(
            "TODO: The `networkx` graph library does not give the right results with use_linked_mask=True."
            "Use `networkit` and `igraph` libraries instead if you would like to set use_linked_mask=True."
        )

    if coords is None and None in (coords1, coords2):
        raise ValueError(
            "either pass `coords` for internal matching or a pair of coordinate lists/arrays"
            "(`coords1` and `coords2`) for cross-matching"
        )
    elif (coords1 is not None and coords2 is not None) and coords is not None:
        raise ValueError(
            "either pass `coords` for internal matching or a pair of coordinate lists/arrays"
            "(`coords1` and `coords2`) for cross-matching"
        )

    if coords is not None:
        num_objects = len(coords)
        if njobs > 1:
            coords_chunks, refidx_chunks, overidx_chunks = get_mosaic_sets(
                coords=coords,
                coords1=None,
                coords2=None,
                linking_length=linking_length,
                wcs=None,
                mode="all",
                nside_mosaics=None,
                njobs=njobs,
                overlap=overlap,
                use_linked_mask=use_linked_mask,
            )
            idxshift_list = [0] + list(np.cumsum([len(x) for x in coords_chunks[:-1]]))
    else:
        raise NotImplementedError('Runs are not defined for when coords is None')  # TODO
        # num_objects = len(coords1) + len(coords2)
        # (
        #     coord1_chunks,
        #     coord2_chunks,
        #     refidx1_chunks,
        #     refidx2_chunks,
        #     overidx1_chunks,
        #     overidx2_chunks,
        # ) = get_mosaic_sets(
        #     coords=None,
        #     coords1=coords1,
        #     coords2=coords2,
        #     linking_length=linking_length,
        #     wcs=None,
        #     mode="all",
        #     nside_mosaics=None,
        #     njobs=njobs,
        #     overlap=overlap,
        #     use_linked_mask=use_linked_mask,
        # )
        # idxshift_list1 = [0] + list(np.cumsum([len(x) for x in coords1_chunks[:-1]]))  # noqa
        # idxshift_list2 = [0] + list(np.cumsum([len(x) for x in coords2_chunks[:-1]])). # noqa

    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    tqdm_kwargs.setdefault("desc", "Creating matching lists")
    tqdm_kwargs.setdefault(
        "bar_format", "{elapsed}|{bar}|{remaining} ({desc}: {percentage:0.0f}%)"
    )

    show_progress = show_progress and session.viewedonscreen()

    kwargs = {
        "linking_length": linking_length,
        "graph_lib": graph_lib,
        "num_threads": num_threads,
        "num_objects": num_objects,
        "verbose": verbose,
        "show_progress": show_progress,
        "silent": silent,
        "tqdm_kwargs": tqdm_kwargs,
    }

    make_partial_group_ids = partial(get_group_ids, **kwargs)

    if coords is not None:
        if njobs > 1:
            args = (
                (
                    coords_chunks[job_num],
                    None,
                    None,
                    idxshift_list[job_num],
                    None,
                    None,
                    overidx_chunks[job_num],
                    None,
                    None,
                    f"kdtree_sky_{job_num}" if storekdtree else False,
                    job_num,
                )
                for job_num in range(njobs)
            )
        else:
            args = (
                coords,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                "kdtree_sky" if storekdtree else False,
                -1,
            )
    else:
        if njobs > 1:
            raise NotImplementedError('Runs are not defined for when coords is None')  # TODO
            # args = (
            #     (
            #         None,
            #         coords1_chunks[i],
            #         coords2_chunks[i],
            #         idxshift_list1[cidx1],
            #         idxshift_list1[cidx1],
            #         None,
            #         overidx_chunks1[job_num],
            #         overidx_chunks2[job_num],
            #         f"kdtree_sky_{cidx1}_{cidx2}" if storekdtree else False,
            #         job_num,
            #     )
            #     for job_num, (cidx1, cidx2) in enumerate(cross_idx)
            # )
        else:
            pass  # ... add here when needed

    if njobs > 1:
        with multiprocessing.Pool(processes=njobs) as pool:
            # It is very crucial that `multiprocessing.Pool` keeps the original
            # order of data passed to `pool.starmap`
            res = pool.starmap(make_partial_group_ids, args)

        with BusyPal(
            "Concatenating the results from different processes",
            style={"id": 6, "color": "sandy_brown"},
            fmt="{spinner} {message}",
        ):
            refidx = np.concatenate(refidx_chunks)
            if use_linked_mask:
                group_ids_chunks = [item[0] for item in res]
                linked_mask_chunks = [item[1] for item in res]
                group_ids = np.concatenate(group_ids_chunks)
                linked_mask = np.concatenate(linked_mask_chunks)
                assert len(group_ids) == len(linked_mask) == len(refidx)
            else:
                group_ids = np.concatenate(res, axis=0)
                assert len(group_ids) == len(refidx)
                linked_mask = None

            del res, refidx_chunks
    else:
        # Serial
        group_ids = make_partial_group_ids(*args)

    # Merge the duplicates and find the underlying network shared between them.
    if njobs > 1:
        # Stitch fragmented group ids from different patches/mosaics.
        if linked_mask is not None:
            with BusyPal(
                "Stitch fragmented group ids from different mosaic sets",
                style={"id": 6, "color": "sandy_brown"},
                fmt="{spinner} {message}",
            ):
                group_ids[linked_mask] = update_labels(
                    refidx[linked_mask], group_ids[linked_mask]
                )
        else:
            with BusyPal(
                "Stitch fragmented group ids from different mosaics - no linked_mask",
                style={"id": 6, "color": "sandy_brown"},
                fmt="{spinner} {message}",
            ):
                group_ids = update_labels(refidx, group_ids)

        # Put the two arrays in a single dataframe.
        df = pd.DataFrame({"idx": refidx, "group_id": group_ids})

        with BusyPal(
            "Rearrange indices to be the same as original input",
            style={"id": 6, "color": "sandy_brown"},
            fmt="{spinner} {message}",
        ):
            # We should drop duplicate idx records from df.
            df.drop_duplicates(subset="idx", keep="first", inplace=True)
            df.sort_values(by="idx", inplace=True)

        assert len(df) == num_objects

        group_ids = df["group_id"]

    if reassign_group_indices:
        group_ids = np.unique(group_ids, return_inverse=True)[1]

    return group_ids
