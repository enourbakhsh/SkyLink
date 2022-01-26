import os
import skylink
from skylink import testing
import numpy as np
from astropy.table import Table
import FoFCatalogMatching
import pytest  # noqa

# TODO: test the matching with more than two catalogs
# TODO: test N-way matching with `linking_lengths` as a dictionary
# TODO: catch illegal footprints due to the gnome projection
# TODO: test MPI implementation
# TODO: test a wide range of linking lengths

graph_lib = "networkit"
ncpus_max = os.cpu_count()  # maximum number of cpus
linking_lengths_default = 0.75  # arcsec
n = 2_000  # number of objects for the mock-up data


def make_mockup():
    def tnormal(mu=None, sigma=None, n=None, lower=-0.5, upper=0.5):
        return np.clip(np.random.normal(np.repeat(mu, n), sigma), lower, upper)

    np.random.seed(2)
    ra = np.random.uniform(4, 6, n)
    dec = np.random.uniform(-1, 1, n)

    cat_a = Table({"ra": ra, "dec": dec})
    cat_b = Table(
        {
            "ra": np.append(ra + tnormal(0, 0.0004, n), ra + tnormal(0, 0.0001, n)),
            "dec": np.append(dec + tnormal(0, 0.0002, n), dec + tnormal(0, 0.0002, n)),
        }
    )

    return cat_a, cat_b


def run_FoFCatalogMatching(cat_a, cat_b, return_pandas=False):
    """ Genetare an output using `FoFCatalogMatching` as our benchmark """
    res_fcm = FoFCatalogMatching.match(
        {"a": cat_a, "b": cat_b}, linking_lengths_default
    )
    if return_pandas:
        return res_fcm.to_pandas()
    else:
        return res_fcm


def test_graph_lib():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl = skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        nprocs=ncpus_max,
        silent=True,
        return_pandas=True,
    )
    testing.assert_equal(res_fcm, res_sl)


def run_with_ncpus(cat_a, cat_b, ncpus):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        nprocs=ncpus,
        silent=True,
        return_pandas=True,
    )


def test_nprocs():
    # TODO: test equality with more than 2 catalogs
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_ncpus(cat_a, cat_b, 1)
    res_sl2 = run_with_ncpus(cat_a, cat_b, 2)
    res_sl3 = run_with_ncpus(cat_a, cat_b, ncpus_max)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)
    testing.assert_equal(res_sl2, res_sl3)


def run_with_overlap(cat_a, cat_b, overlap):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        overlap=overlap,
        nprocs=ncpus_max,
        silent=True,
        return_pandas=True,
    )


def test_overlap():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_overlap(cat_a, cat_b, 1.0)
    res_sl2 = run_with_overlap(cat_a, cat_b, 1.1)
    res_sl3 = run_with_overlap(cat_a, cat_b, 1.2)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)
    testing.assert_equal(res_sl2, res_sl3)


def run_with_linked_mask(cat_a, cat_b, use_linked_mask):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        use_linked_mask=use_linked_mask,
        nprocs=ncpus_max,
        silent=True,
        return_pandas=True,
    )


@pytest.mark.skip(
    reason="FIXME: The `networkx` graph library does not give the right results with use_linked_mask=True"
)
def test_linked_mask():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_linked_mask(cat_a, cat_b, True)
    res_sl2 = run_with_linked_mask(cat_a, cat_b, False)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)


def test_cat_orders():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_overlap(cat_a, cat_b, 1.0)
    res_sl2 = run_with_overlap(cat_b, cat_a, 1.0)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)


def run_with_sort(cat_a, cat_b, sort):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        sort=sort,
        nprocs=ncpus_max,
        silent=True,
        return_pandas=True,
    )


def test_sort():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_sort(cat_a, cat_b, True)
    res_sl2 = run_with_sort(cat_b, cat_a, False)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)


def run_with_num_threads(cat_a, cat_b, num_threads):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        nprocs=ncpus_max // num_threads,
        num_threads=num_threads,
        silent=True,
        return_pandas=True,
    )


def test_num_threads():
    # TODO: test equality with more than 2 catalogs
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl1 = run_with_num_threads(cat_a, cat_b, 1)
    res_sl2 = run_with_num_threads(cat_a, cat_b, 2)
    res_sl3 = run_with_num_threads(cat_a, cat_b, ncpus_max)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)
    testing.assert_equal(res_sl2, res_sl3)


def run_with_storekdtree(cat_a, cat_b, storekdtree):
    return skylink.match(
        {"a": cat_a, "b": cat_b},
        linking_lengths=linking_lengths_default,
        graph_lib=graph_lib,
        storekdtree=storekdtree,
        nprocs=ncpus_max,
        silent=True,
        return_pandas=True,
    )


def test_storekdtree():
    cat_a, cat_b = make_mockup()
    res_fcm = run_FoFCatalogMatching(cat_a, cat_b, return_pandas=True)
    res_sl2 = run_with_storekdtree(cat_b, cat_a, False)
    res_sl1 = run_with_storekdtree(cat_a, cat_b, True)
    testing.assert_equal(res_fcm, res_sl1)
    testing.assert_equal(res_sl1, res_sl2)
