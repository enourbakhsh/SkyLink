# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module contains functions for matching coordinate catalogs.
-------------------------------------------------------------------------
It is transferred from astropy v4.0.1 for some modifications by E.N.:
(1) Constructing only one tree if coords1==coords2 in search_around_sky()
(2) Added environment-aware `tqdm` progressbars (js or ascii)
(3) Added verbose
(4) Added busy indicators
"""

import numpy as np
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy import units as u
from astropy.coordinates import Angle
from busypal import BusyPal, session

# from .. import session
import colored as cl


def dummy_tqdm(iterable, **tqdm_kwargs):  # Dummy tqdm
    return iterable


if session.javascript_friendly():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

__all__ = ["search_around_sky"]


def search_around_sky(
    coords=None,
    coords1=None,
    coords2=None,
    seplimit=None,
    storekdtree="kdtree_sky",
    verbose=1,
    show_progress=True,
    silent=False,
    return_dists=False,
    tqdm_kwargs={},
):
    """Searches for pairs of points that have an angular separation at least as
    close as a specified angle.

    This is intended for use on coordinate objects with arrays of coordinates,
    not scalars. For scalar coordinates, it is better to use the ``separation``
    methods.

    Parameters
    ----------
    coords1 : `~astropy.coordinates.BaseCoordinateFrame` or
        `~astropy.coordinates.SkyCoord`
        The first set of coordinates, which will be searched for matches from
        ``coords2`` within ``seplimit``. Cannot be a scalar coordinate.
    coords2 : `~astropy.coordinates.BaseCoordinateFrame` or
        `~astropy.coordinates.SkyCoord`
        The second set of coordinates, which will be searched for matches from
        ``coords1`` within ``seplimit``. Cannot be a scalar coordinate.
    seplimit : `~astropy.units.Quantity` with angle units
        The on-sky separation to search within.
    storekdtree : bool or str, optional
        If a string, will store the KD-Tree used in the search with the name
        ``storekdtree`` in ``coords2.cache``. This speeds up subsequent calls
        to this function. If False, the KD-Trees are not saved.
    verbose : int, optional
        If 0, it does not write any words to stdout while running (it will
        still show the progressbars and busy indicators whenever needed)
        If 1, it writes some extra information about the status of the job
    show_progress : bool, optional
        If True, it will show the progressbars and busy indicators whenever
        needed. If False, it hides them. Except, in the case of busy indicators
        it will only hide the animation part not the message.
    silent : bool, optional
        If True, it supresses the stdout, i.e. it sets `verbose=0` and
        `show_progress=False` plus it hides the busy indicators alltogether
        along with any message they might have.
        `silent` overrides `verbose` and `show_progress`.
    return_dists : bool, optional
        If True, it returns sep2d and dist3d in addition to the matching
        indices (i.e. idx1, idx2)

    Returns
    -------
    idx1 : integer array
        Indices into ``coords1`` that matches to the corresponding element of
        ``idx2``. Shape matches ``idx2``.
    idx2 : integer array
        Indices into ``coords2`` that matches to the corresponding element of
        ``idx1``. Shape matches ``idx1``.
    sep2d : `~astropy.coordinates.Angle`
        The on-sky separation between the coordinates. Shape matches ``idx1``
        and ``idx2``.
    dist3d : `~astropy.units.Quantity`
        The 3D distance between the coordinates. Shape matches ``idx1``
        and ``idx2``; the unit is that of ``coords1``.
        If either ``coords1`` or ``coords2`` don't have a distance,
        this is the 3D distance on the unit sphere, rather than a
        physical distance.

    Notes
    -----
    This function requires `SciPy <https://www.scipy.org/>`_ (>=0.12.0)
    to be installed or it will fail.

    In the current implementation, the return values are always sorted in the
    same order as the ``coords1`` (so ``idx1`` is in ascending order).  This is
    considered an implementation detail, though, so it could change in a future
    release.
    """

    if show_progress:
        skip_busypal = -1
        disable_tqdm = False
    else:
        skip_busypal = 1
        disable_tqdm = True

    if silent:
        verbose = 0
        skip_busypal = 2
        disable_tqdm = True

    if coords is not None:
        if coords1 is not None or coords2 is not None:
            raise ValueError(
                "either use `coords` for internal matching or `coords1` and \
                `coords2` for cross matching."
            )
        coords1 = coords
        coords2 = coords

    if not seplimit.isscalar:
        raise ValueError("seplimit must be a scalar in search_around_sky")

    if coords1.isscalar or coords2.isscalar:
        raise ValueError(
            "One of the inputs to search_around_sky is a scalar. "
            "search_around_sky is intended for use with array "
            "coordinates, not scalars.  Instead, use "
            "``coord1.separation(coord2) < seplimit`` to find the "
            "coordinates near a scalar coordinate."
        )

    if len(coords1) == 0 or len(coords2) == 0:
        # Empty array input: return empty match
        if coords2.distance.unit == u.dimensionless_unscaled:
            distunit = u.dimensionless_unscaled
        else:
            distunit = coords1.distance.unit
        if return_dists:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                Angle([], u.deg),
                u.Quantity([], distunit),
            )
        else:
            return (np.array([], dtype=int), np.array([], dtype=int))

    if all(coords1 == coords2):
        same_coords = True
    else:
        # We convert coord1 to match coord2's frame.  We do it this way
        # so that if the conversion does happen, the KD tree of coord2 at least
        # gets saved.
        # (by convention, coord2 is the "catalog" if that makes sense)
        coords1 = coords1.transform_to(coords2)
        same_coords = False

    # Strip out distance info.
    urepr1 = coords1.data.represent_as(UnitSphericalRepresentation)
    ucoords1 = coords1.realize_frame(urepr1)

    with BusyPal(
        "Growing the first KD-Tree",
        style={"id": 6, "color": "sandy_brown"},
        fmt="{spinner} {message}",
        skip=skip_busypal,
    ):
        kdt1 = _get_cartesian_kdtree(ucoords1, storekdtree)

    if storekdtree and coords2.cache.get(storekdtree):
        # Just use the stored KD-Tree.
        kdt2 = coords2.cache[storekdtree]
    else:
        if same_coords:
            kdt2 = kdt1
            if verbose:
                print(
                    cl.stylize("✔", cl.fg("green") + cl.attr("bold")) + " Used the same KD-Tree for the"
                    "second set of coordinates since this is an internal match."
                )
        else:
            # Strip out distance info.
            urepr2 = coords2.data.represent_as(UnitSphericalRepresentation)
            ucoords2 = coords2.realize_frame(urepr2)
            with BusyPal(
                "Growing the second KD-Tree",
                style={"id": 6, "color": "SANDY_BROWN"},
                fmt="{spinner} {message}",
                skip=skip_busypal,
            ):
                kdt2 = _get_cartesian_kdtree(ucoords2, storekdtree)
        if storekdtree:
            # Save the KD-Tree in coords2, *not* ucoords2.
            coords2.cache["kdtree" if storekdtree is True else storekdtree] = kdt2
            if verbose:
                print(
                    cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
                    + " Stored the KD-Tree in the cache."
                )

    # This is the *cartesian* 3D distance that corresponds to the given angle.
    r = (2 * np.sin(Angle(seplimit) / 2.0)).value

    idxs1 = []
    idxs2 = []

    with BusyPal(
        f"Finding all pairs of points whose distance is at most {seplimit}",
        style={"id": 6, "color": "sandy_brown"},
        fmt="{spinner} {message}",
        skip=skip_busypal,
    ):
        found_pairs = kdt1.query_ball_tree(kdt2, r)

    for i, matches in enumerate(tqdm(found_pairs, disable=disable_tqdm, **tqdm_kwargs)):
        for match in matches:
            idxs1.append(i)
            idxs2.append(match)
    if verbose:
        print(
            "\r\r"
            + cl.stylize("✔", cl.fg("green") + cl.attr("bold"))
            + " Created matching lists from KD-Trees"
        )
    idxs1 = np.array(idxs1, dtype=int)
    idxs2 = np.array(idxs2, dtype=int)

    if return_dists:
        if idxs1.size == 0:
            if coords2.distance.unit == u.dimensionless_unscaled:
                distunit = u.dimensionless_unscaled
            else:
                distunit = coords1.distance.unit
            d2ds = Angle([], u.deg)
            d3ds = u.Quantity([], distunit)
        else:
            d2ds = coords1[idxs1].separation(coords2[idxs2])
            try:
                d3ds = coords1[idxs1].separation_3d(coords2[idxs2])
            except ValueError:
                # They don't have distances, so we just fall back on the
                # cartesian distance, computed from d2ds.
                d3ds = 2 * np.sin(d2ds / 2.0)
        return idxs1, idxs2, d2ds, d3ds
    else:
        return idxs1, idxs2


def _get_cartesian_kdtree(coord, attrname_or_kdt="kdtree", forceunit=None):
    """
    This is a utility function to retrieve (and build/cache, if necessary)
    a 3D cartesian KD-Tree from various sorts of astropy coordinate objects.

    Parameters
    ----------
    coord : `~astropy.coordinates.BaseCoordinateFrame` or
        `~astropy.coordinates.SkyCoord`
        The coordinates to build the KD-Tree for.
    attrname_or_kdt : bool or str or KDTree
        If a string, will store the KD-Tree used for the computation in the
        ``coord``, in ``coord.cache`` with the provided name. If given as a
        KD-Tree, it will just be used directly.
    forceunit : unit or None
        If a unit, the cartesian coordinates will convert to that unit before
        being put in the KD-Tree.  If None, whatever unit it's already in
        will be used

    Returns
    -------
    kdt : `~scipy.spatial.cKDTree` or `~scipy.spatial.KDTree`
        The KD-Tree representing the 3D cartesian representation of the input
        coordinates.
    """
    from warnings import warn

    # Without scipy this will immediately fail.
    from scipy import spatial

    try:
        KDTree = spatial.cKDTree
    except Exception:
        warn(
            "C-based KD tree not found, falling back on (much slower) "
            "python implementation"
        )
        KDTree = spatial.KDTree

    if attrname_or_kdt is True:  # Backwards compatibility for pre v0.4
        attrname_or_kdt = "kdtree"

    # Figure out where any cached KDTree might be.
    if isinstance(attrname_or_kdt, str):
        kdt = coord.cache.get(attrname_or_kdt, None)
        if kdt is not None and not isinstance(kdt, KDTree):
            raise TypeError(
                f'The `attrname_or_kdt` "{attrname_or_kdt}" is not a scipy KD tree!'
            )
    elif isinstance(attrname_or_kdt, KDTree):
        kdt = attrname_or_kdt
        attrname_or_kdt = None
    elif not attrname_or_kdt:
        kdt = None
    else:
        raise TypeError(
            "Invalid `attrname_or_kdt` argument for KD-Tree:" + str(attrname_or_kdt)
        )

    if kdt is None:
        # Need to build the cartesian KD-tree for the catalog.
        if forceunit is None:
            cartxyz = coord.cartesian.xyz
        else:
            cartxyz = coord.cartesian.xyz.to(forceunit)
        flatxyz = cartxyz.reshape((3, np.prod(cartxyz.shape) // 3))
        # There should be no NaNs in the kdtree data.
        if np.isnan(flatxyz.value).any():
            raise ValueError("Catalog coordinates cannot contain NaN entries.")
        try:
            # Set compact_nodes=False, balanced_tree=False to use
            # "sliding midpoint" rule, which is much faster than standard for
            # many common use cases
            kdt = KDTree(flatxyz.value.T, compact_nodes=False, balanced_tree=False)
        except TypeError:
            # Python implementation does not take compact_nodes and
            # balanced_tree as arguments. However, it uses sliding midpoint
            # rule by default
            kdt = KDTree(flatxyz.value.T)

    if attrname_or_kdt:
        # cache the kdtree in `coord`
        coord.cache[attrname_or_kdt] = kdt

    return kdt
