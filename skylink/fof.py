import sys
import numpy as np
from .astropy_search.matching import search_around_sky
from .graph import GraphDataStructure
from astropy.coordinates import SkyCoord, Angle
from itertools import accumulate, chain, product
import datetime
from functools import partial
import multiprocessing
from busypal import busy, BusyPal, session
import pandas as pd
from collections import Counter
import colored as cl

# TODO: MPI - almost there in `skylink.py`!!
# FIXME: progressbars and busy indicators (in particular the ones that use decorators) still show up with `silent=True`

# Complex Network Analysis: The Need for Speed (Benchmark Paper)
# http://m3nets.de/publications/CCC2016d.pdf

# networkit parallelism
# nk.setNumberOfThreads(8) # set the maximum number of available threads
# nk.getMaxNumberOfThreads() # see maximum number of available threads
# nk.getCurrentNumberOfThreads() # the number of threads currently executing

# try with 100 or smaller - sometimes (rarely) it hangs eternally!

__all__ = ['fastmatch']

def update_labels(items, labels): # stackoverflow... find the link!
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

# TODO: also make a function to stitch overlapping patches with pre-calculated group_ids in general
def stitch_group_ids(items, labels, graph_lib='networkit', verbose=True, num_threads=None):

    labels_max = labels.max() # max() was slower!
    items = items+labels_max+1 # we somehow mark them since they need to be removed from clustered integers later on #+labels_max+1 #map(str, items)
    edges = zip(items,labels)

    nnodes = items.max()+labels_max+1 if graph_lib=='networkit' else None # not needed otherwise
    gds = GraphDataStructure(graph_lib, num_threads=num_threads) # num_threads=4 # for networkit only
    graph = gds.build_graph_from_edges(edges, verbose=verbose, nnodes=nnodes)
    del edges
    cc = find_clusters(graph, graph_lib=graph_lib, verbose=False) #verbose)
     
    a = list(map(list, (x for x in cc if len(x)>2))) #list(map(sorted, A))
    del cc
    a = [[w for w in x if w<=labels_max] for x in a] # slowest!! #[[w for w in x if not isinstance(w, str)] for x in a]

    b = [[x[0]]*(len(x)-1) for x in a]
    for x in a:
        del x[0]

    a = list(chain(*a)) # orders of mag faster than np.concatenate(a) in this case with lists
    b = list(chain(*b))

    mapping = np.arange(0,labels_max+1)
    mapping[a] = b
    new_labels = mapping[labels]
    return new_labels

## slow:
# def stitch_group_ids(items, labels, graph_lib='networkx', verbose=True, num_threads=None, nnodes=None):
    
#     items = map(str, items)
#     edges = zip(items,labels)
#     # G=nx.Graph(edges)
#     # cc = nx.connected_components(G)
    
#     gds = GraphDataStructure(graph_lib, num_threads=num_threads) # num_threads=4 # for networkit only
#     graph = gds.build_graph_from_edges(edges, verbose=verbose, nnodes=nnodes)
#     cc = find_clusters(graph, graph_lib=graph_lib, verbose=verbose)
     
#     a = (x for x in cc if len(x)>2)
#     a = map(list, a)
#     a = [[w for w in x if not isinstance(w, str)] for x in a]

#     b = [[x[0]]*(len(x)-1) for x in a]
#     for x in a:
#         del x[0]

#     a = list(chain(*a))
#     b = list(chain(*b))

#     mapping = np.arange(0,max(labels)+1)
#     mapping[a] = b
#     new_labels = mapping[labels]
#     return new_labels


# https://python.hotexamples.com/examples/astropy.coordinates/SkyCoord/to_pixel/python-skycoord-to_pixel-method-examples.html   
def radec2xy(coords, ra_unit='deg', dec_unit='deg', wcs=None, mode='all', pixel_scale=1):
    """Project a catalog onto the image plane.

    The default is the tangent image plane, but any WCS transformation
    can be used.

    Parameters
    ----------
    cat : astropy SkyCoord, Quantity, or Angle
      The coordinate list.  If a `Quantity` or `Angle`, `cat` must be
      a 2xN array: (lat, long).
    wcs : astropy.wcs.WCS, optional
      The world coordinate system object for transformation.  The
      default assumes the tangent plane centered on the latitude
      and longitude of the catalog.
    mode : string, optional
      The projection mode for `SkyCoord` objects: 'wcs' or 'all'.  See
      `SkyCoord.to_pixel`.

    Returns
    -------
    flat_cat : ndarray
      A 2xN array of pixel positions, (x, y).

    zero-indexed pixel coord because of (0,0)

    """

    # https://github.com/astropy/astropy/issues/2847
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    if not isinstance(coords, SkyCoord):
        assert len(coords) == 2
        coords = SkyCoord(ra=coords[0], dec=coords[1], unit=(ra_unit,dec_unit))

    # - create the projection without having to deal with FITS files
    if wcs is None:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = (0, 0)                     # pixel coordinate of the reference point --> the projection is centered at 0, 0
        ra_tan = (coords.ra.max() - coords.ra.min()) / 2.0 + coords.ra.min()
        dec_tan = (coords.dec.max() - coords.dec.min()) / 2.0 + coords.dec.min()
        wcs.wcs.crval = (ra_tan.value, dec_tan.value)    # coordinate value at reference point
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]   # axis type
        wcs.wcs.cdelt = [pixel_scale/3600., pixel_scale/3600.]  # coordinate increment > for pixel_scale=1, The pixel scale is 1"/pixel, this outputs values in arcsecond offset

    x, y = coords.to_pixel(wcs, mode=mode)

    # x, y = wcs.wcs_world2pix(coords.ra,coords.dec,0)
    if (~np.isfinite(x)).sum()!=0:
        raise ValueError('We encountered one or more non-finite values in the projected coordinates.')
        
    if (~np.isfinite(x)).sum()==len(x):
        raise ValueError('The projection led to all non-finite pixel coordinates, probably because it did not converge. Make sure your ra and dec ranges are valid (0<ra<360 and -45<dec<45 in degrees) and gnomic-projectable onto a plane. Also see https://docs.astropy.org/en/stable/_modules/astropy/wcs/wcs.html')

    return x, y #np.vstack((x, y))


# tqdm automatically switches to the text-based
# progress bar if not running in Jupyter
try: # https://github.com/tqdm/tqdm/issues/506
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:  # jupyter
        from tqdm.notebook import tqdm
        # # https://github.com/bstriner/keras-tqdm/issues/21
        # # this removes the vertical whitespace that remains when tqdm progressbar disappears
        # from IPython.core.display import HTML
        # HTML("""
        # <style>
        # .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
        #   padding: 0;
        #   border: 0;
        # }
        # </style>
        # """)
    if 'terminal' in ipy_str:  # ipython
        from tqdm import tqdm 
except:                        # terminal
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable

# def make_graph(coords=None, coords1=None,coords2=None,coords1_idxshift=None,coords2_idxshift=None,storekdtree=None,job_num=0,linking_length=None,graph_lib='igraph', verbose=1, show_progress=True, silent=False, tqdm_kwargs={}):
#     # coord1 should not be partial!!
#     # coord2 can be partial
#     # !!! good thing about networkx: does not make unnecessary isolated ---> faster performance at least in the serial mode
#     # graph_lib: 'igraph', 'igraph', 'networkit'
#     # networkx does not make unnecessary isolated nodes, the other two do
#     # so networkx is better for when we expect many isolated coords in a huge dataset
#     # however, networkx is purely in python, the other two are in c and networkit has the added benefit of openMP!
#     # igraph and networkx add the edges at onece but networkit needs to use a loop to add them one by one
#     # (networkit's c and R version has addlist that does it at once but couldn't find the same functionality in tehir python version)
#     # check the new versions of these libraries to see if they added more capabilities
#     # if coords1==coords2: print('same....')
#     t0 = datetime.datetime.now()
#     tqdm_kwargs['position'] = job_num

#     if silent:
#         verbose=0

#     # if job_num!=0:
#     #     verbose=False
#     #     silent=True

#     idx1, idx2 = search_around_sky(coords1, coords2, Angle(f'{linking_length}s'), storekdtree=storekdtree, verbose=verbose, show_progress=show_progress, silent=silent, tqdm_kwargs=tqdm_kwargs) #coords1.search_around_sky(coords2, Angle(f'{linking_length}s'))

#     if verbose:
#         print(f'kdtree done in about {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')

#     idx1 += coords1_idxshift
#     idx2 += coords2_idxshift
#     graph_edges = set((a,b) if a<b else (b,a) for a,b in zip(idx1, idx2) if a!=b) # delete duplicates and 1:1 singles (It is important that a<b is enforced for networkit nnodes finder to work)
#     # gds = GraphDataStructure(graph_lib) # num_threads=4 # for networkit only
#     # graph = gds.build_graph_from_edges(graph_edges, verbose=verbose)

#     return graph_edges

def get_group_ids(coords=None, coords1=None,coords2=None, coords_idxshift=None, coords1_idxshift=None, coords2_idxshift=None, overidx=None, overidx1=None, overidx2=None, storekdtree=None,job_num=0,linking_length=None,graph_lib='igraph', num_threads=None, num_objects=None, verbose=1, show_progress=True, silent=False, tqdm_kwargs={}):
    # coord1 should not be partial!!
    # coord2 can be partial
    # !!! good thing about networkx: does not make unnecessary isolated ---> faster performance at least in the serial mode
    # graph_lib: 'igraph', 'igraph', 'networkit'
    # networkx does not make unnecessary isolated nodes, the other two do
    # so networkx is better for when we expect many isolated coords in a huge dataset
    # however, networkx is purely in python, the other two are in c and networkit has the added benefit of openMP!
    # igraph and networkx add the edges at onece but networkit needs to use a loop to add them one by one
    # (networkit's c and R version has addlist that does it at once but couldn't find the same functionality in tehir python version)
    # check the new versions of these libraries to see if they added more capabilities
    # if coords1==coords2: print('same....')
    t0 = datetime.datetime.now()
    
    if job_num==-1:
        job_num = 0
        parallel = False
    else:
        parallel = True

    tqdm_kwargs['position'] = job_num

    # if job_num!=0:
    #     verbose=False
    #     silent=True

    if job_num != 0:
        silent = True

    skip_busypal = -1 if show_progress else 1

    if silent:
        verbose=0
        skip_busypal = 2
        disable_tqdm = True
    else:
        disable_tqdm = False




    # print('session.viewedonscreen()',session.viewedonscreen())

    idx1, idx2 = search_around_sky(coords=coords, coords1=coords1, coords2=coords2, seplimit=Angle(f'{linking_length}s'), storekdtree=storekdtree, verbose=verbose, show_progress=show_progress, silent=silent, tqdm_kwargs=tqdm_kwargs) #coords1.search_around_sky(coords2, Angle(f'{linking_length}s'))

    if verbose:
        pass
        # print(f'kdtree done in about {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')

        # multoprocessing has barrier as well:
        # 'waiting for all processes to catch up'


    # idx1 += coords1_idxshift
    # idx2 += coords2_idxshift

    graph_edges = set((a,b) if a<b else (b,a) for a,b in zip(idx1, idx2) if a!=b) # delete duplicates and 1:1 singles (It is important that a<b is enforced for networkit nnodes finder to work)
    # gds = GraphDataStructure(graph_lib) # num_threads=4 # for networkit only
    # graph = gds.build_graph_from_edges(graph_edges, verbose=verbose)

    num_objects_chunk = len(coords) if coords is not None else len(coords1)+len(coord2)

    with BusyPal('Building the representative graph/network', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}', skip=skip_busypal):
        gds = GraphDataStructure(graph_lib, num_threads=num_threads) # threads are for networkit only
        #with BusyPal(f'Building the graph using the {graph_lib} library'):
        final_graph = gds.build_graph_from_edges(graph_edges, verbose=False, nnodes=num_objects_chunk)
        clusters = find_clusters(final_graph, graph_lib=graph_lib, verbose=False)
    
    nclusters = len(clusters) #max(chain.from_iterable(seq))
    starting_id = (job_num+2)*num_objects+coords_idxshift # make them very far from each other (absolutely no chance of a conflict)
    group_ids = np.arange(starting_id, starting_id+num_objects_chunk)
    linked_mask = np.zeros(num_objects_chunk, dtype=bool)
    
    del tqdm_kwargs['desc']
    if overidx is not None:
        overidx = set(overidx) # makes the lookups very fast!
        for idx, cluster in enumerate(tqdm(clusters, total=nclusters, desc='Finding connected components of the'+'graphs and shared components' if parallel else 'graph', disable=disable_tqdm, **tqdm_kwargs)):
            if any(gidx in overidx for gidx in cluster): # if any of the connected components has a foot on the overlap region that whole group should be involved in stitching later
                # print('yes!')
                linked_mask[cluster] = True
            group_ids[cluster] = idx+coords_idxshift
            # for galaxy_idx in cluster:
            #     group_ids[galaxy_idx] = idx+coords_idxshift
        del clusters
        if verbose:
            print('\r\r'+cl.stylize('✔', cl.fg('green')+cl.attr('bold'))+' Assigned group ids for each chunk by using connected components of the graphs' if parallel else 'by using connected components of the graph')
        return group_ids, linked_mask
    else: # it might be parallel but it is not using linked_mask  
        for idx, cluster in enumerate(tqdm(clusters, total=nclusters, desc='Finding connected components of the'+'graphs' if parallel else 'graph', disable=disable_tqdm, **tqdm_kwargs)):
            group_ids[cluster] = idx+coords_idxshift
            # for galaxy_idx in cluster:
            #     group_ids[galaxy_idx] = idx+coords_idxshift
        del clusters
        if verbose:
            print('\r\r'+cl.stylize('✔', cl.fg('green')+cl.attr('bold'))+' Assigned group ids for each chunk' if parallel else ' Assigned group ids')
        return group_ids

    # idx_isolated = (group_ids==-1)
    # with BusyPal('np.arange'):
    # group_ids[idx_isolated] = np.arange(nclusters, nclusters+idx_isolated.sum())
    # print('***',group_ids)
    # return group_ids, linked_mask

# @busy('Clustering')
def find_clusters(graphs,graph_lib='igraph',verbose=True):
    # graphs can be a list/tuple or just a single graph
    t0 = datetime.datetime.now()
    gds = GraphDataStructure(graph_lib) # num_threads=4 # for networkit only
    # - merge and cluster (it does the merging internally if you pass a list or tuple of graphs)
    clusters = gds.cluster(graphs,verbose=verbose)
    if verbose:
        print(f'clustering done in {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')
    # - remove isolated points
    # t0 = datetime.datetime.now()
    # clusters = list(filter(lambda x: len(x)>1, clusters))
    # clusters = [sl for sl in clusters if len(sl)>1] # not as elegant
    # print(f'removing isolated clusters for {graph_lib} generated cluster done in {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')
    return clusters

# # modified from https://stackoverflow.com/questions/56120273/quicker-way-to-implement-numpy-isin-followed-by-sum
# def fast_isin_int(A,a): # at least 2X faster than np.isin for integers (numpy might make it's np.isin ~10X faster in the future, watch its github)
#     # suitable for arrays containing small integers like less than 1e7 
#     grid = np.zeros(max(np.max(A),np.max(a))+1, bool)
#     grid[a] = True
#     return grid[A]

@busy('Mosaicking data', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}') #TODO: make it paralllel!!! change plural masoiacs to mosaic -- since mosaic is plural by default!
def get_mosaic_sets(coords=None, coords1=None, coords2=None, linking_length=None, wcs=None, mode='all', nside_mosaics=None, njobs=None, overlap=1.0, use_linked_mask=False):
    # nside_mosaics : resolution parameter
    # print(coords1,coords2,coords)
    # if coords1==coords2==None:
    #     print('ggg')
    # assert (coords is not None and not (coords1==coords2==None))
    # Note: x and y are projection of ra and dec onto the image plane (temporarilly made them for splitting purpose)
    x, y = radec2xy(coords, wcs=wcs, mode=mode)

    if nside_mosaics is None:
        nside_mosaics = int(2*np.sqrt(njobs)) # a func of njobs somehow!! this way each job takes care of multiple mosaics
    
    H, xedges, yedges = np.histogram2d(x, y, bins=nside_mosaics)
    idx_filled_mosaics = np.where(H>0)
    num_filled_mosaics = len(idx_filled_mosaics[0])
    xbounds, ybounds, refidx_inside, refidx_overlap = [], [], [], []

    for idx_x, idx_y in zip(*idx_filled_mosaics):
        xbounds.append([xedges[idx_x], xedges[idx_x+1]])
        ybounds.append([yedges[idx_y], yedges[idx_y+1]])

    for xbound, ybound in zip(xbounds, ybounds):
        # - create overlapping mosaics?
        #   if yes, technically 0.5*l (overlap=1) leads to one linking length overlap, but the user can choose to be more conservative
        x0 = xbound[0]-linking_length*float(overlap)/2.0
        x1 = xbound[1]+linking_length*float(overlap)/2.0
        y0 = ybound[0]-linking_length*float(overlap)/2.0
        y1 = ybound[1]+linking_length*float(overlap)/2.0        
        cx0 = x>=x0
        cx1 = x<=x1 if x1==xedges[-1] else x<x1 # x < x1 is enough if overlapping_mosaics = False
        cy0 = y>=y0
        cy1 = y<=y1 if y1==yedges[-1] else y<y1 # y < y1 is enough if overlapping_mosaics = False
        refidx_inside.append(np.where(cx0 & cx1 & cy0 & cy1)[0])
        # idx_inside = np.where(cx0 & cx1 & cy0 & cy1) # integers not bools
        #coords_mosaics.append(coords[idx_inside])

        if use_linked_mask:
            # find the internal bounds it makes with other overlapping mosaics (used for stitching the group id chunks later after concatenating results from different procs)
            x0_p = xbound[0]+linking_length*float(overlap)/2.0
            x1_p = xbound[1]-linking_length*float(overlap)/2.0
            y0_p = ybound[0]+linking_length*float(overlap)/2.0
            y1_p = ybound[1]-linking_length*float(overlap)/2.0
            cx0_p = x<=x0_p #(x0<=x) & (x<=x0_p)
            cx1_p = x1_p<=x #(x1_p<=x) & (x<=x1) if x1==xedges[-1] else (x1_p<=x) & (x<x1) # x < x1 is enough if overlapping_mosaics = False
            cy0_p = y<=y0_p #(y0<=y) & (y<=y0_p)
            cy1_p = y1_p<=y #(y1_p<=y) & (y<=y1) if y1==yedges[-1] else (y1_p<=y) & (y<y1) # y < y1 is enough if overlapping_mosaics = False
            refidx_overlap.append(np.where( (cx0 & cx1 & cy0 & cy1) & (cx0_p | cx1_p | cy0_p | cy1_p) )[0])

    idx_mosaics_for_chunks = np.array_split(range(num_filled_mosaics), njobs)

    # chunks can consist of one or more mosaics which are assigned to each job
    coords_chunks, refidx_chunks = [], []
    overidx_chunks = [] if use_linked_mask else [None]*njobs
    for idx_mosaics in idx_mosaics_for_chunks:
        refidx_inside_unified = np.array([], dtype=np.int64)
        if use_linked_mask:
            refidx_overlap_unified = np.array([], dtype=np.int64)
        for m in idx_mosaics:
            refidx_inside_unified = np.append(refidx_inside_unified, refidx_inside[m])
            if use_linked_mask:
                refidx_overlap_unified = np.append(refidx_overlap_unified, refidx_overlap[m])
        refidx_inside_unified = np.unique(refidx_inside_unified) # there might be some duplicates since little mosaics have overlaps
        coords_chunks.append(coords[refidx_inside_unified])
        refidx_chunks.append(refidx_inside_unified)
        if use_linked_mask:
            refidx_overlap_unified = np.unique(refidx_overlap_unified) # there might be some duplicates since little mosaics have overlaps
            idx_overlap_unified = np.where(np.isin(refidx_inside_unified, refidx_overlap_unified))[0] # gives indices to the chunk array (not the ref catalog)
            overidx_chunks.append(idx_overlap_unified)

    # refidx_chunks # indices to the main catalog

    if coords is not None:
        return coords_chunks, refidx_chunks, overidx_chunks # refidx indices to the main catalog
    else:
        return coords1_chunks, coords2_chunks, refidx_chunks, overidx_chunks #..... FIXME!


def fastmatch(coords=None, coords1=None, coords2=None,linking_length=None, periodic_box_size=None,
              reassign_group_indices=True, njobs=1, overlap=1.0, graph_lib='igraph',
              num_threads=None, storekdtree=True, use_linked_mask=False,
              verbose=1, show_progress=True, silent=False, **tqdm_kwargs):
    '''
    use_linked_mask: bool
        An experimental feature that generates a mask to be applied to the arrays before stitching.
        This reduces the time to create a graph in strich_group_ids() but might have some amount of overhead
        (it can be negligible or a bit significant depending on the data) while making the mask through
        get_mosaics() and get_group_ids(). Experiment it with your data.
    overlap: 1 should be enough to compensate for lost pairs that cross the boundaries (or maybe 1.01 just in case).
    '''

    # - define aliass for graph libraries names
    if graph_lib=='nk':
        graph_lib='networkit'
    elif graph_lib=='nx':
        graph_lib='networkx'
    elif graph_lib=='ig':
        graph_lib='igraph'
    
    if use_linked_mask and graph_lib=='networkx':
        raise ValueError('TODO: The `networkx` graph library does not give the right results with use_linked_mask=True. Use `networkit` and `igraph` libraries instead if you would like to set use_linked_mask=True.')

    # if num_threads is None:
    #     num_threads = njobs

    if coords is None and None in (coords1, coords2):
        raise ValueError('either pass `coords` for internal matching or a pair of coordinate lists/arrays (`coords1` and `coords2`) for cross-matching')
    elif (coords1 is not None and coords2 is not None) and coords is not None:
        raise ValueError('either pass `coords` for internal matching or a pair of coordinate lists/arrays (`coords1` and `coords2`) for cross-matching')

    # --- --- --- ---

    if coords is not None:
        num_objects = len(coords)
        if njobs>1:
            coords_chunks, refidx_chunks, overidx_chunks = get_mosaic_sets(coords=coords, coords1=None, coords2=None, linking_length=linking_length, wcs=None, mode='all', nside_mosaics=None, njobs=njobs, overlap=overlap, use_linked_mask=use_linked_mask)
            idxshift_list = [0]+list(np.cumsum([len(x) for x in coords_chunks[:-1]]))
    else:
        num_objects = len(coords1)+len(coords2)
        coord1_chunks, coord2_chunks, refidx1_chunks, refidx2_chunks, overidx1_chunks, overidx2_chunks = get_mosaic_sets(coords=None, coords1=coords1, coords2=coords2, linking_length=linking_length, wcs=None, mode='all', nside_mosaics=None, njobs=njobs, overlap=overlap, use_linked_mask=use_linked_mask)
        idxshift_list1 = [0]+list(np.cumsum([len(x) for x in coords1_chunks[:-1]]))
        idxshift_list2 = [0]+list(np.cumsum([len(x) for x in coords2_chunks[:-1]]))

    # --- --- --- ---

    if tqdm_kwargs is None:
        tqdm_kwargs={}
    tqdm_kwargs.setdefault('desc', 'Creating matching lists');
    tqdm_kwargs.setdefault('bar_format', '{elapsed}|{bar}|{remaining} ({desc}: {percentage:0.0f}%)')

    show_progress = show_progress and session.viewedonscreen()

    kwargs = {'linking_length':linking_length, 'graph_lib':graph_lib, 'num_threads': num_threads, 'num_objects': num_objects,
              'verbose':verbose, 'show_progress':show_progress, 'silent':silent, 'tqdm_kwargs': tqdm_kwargs}
    make_partial_group_ids = partial(get_group_ids, **kwargs)

    if coords is not None:
        if njobs>1:
            args = ((coords_chunks[job_num], None, None, idxshift_list[job_num], None, None, overidx_chunks[job_num], None, None,  f'kdtree_sky_{job_num}' if storekdtree else False, job_num) for job_num in range(njobs))
        else:
            args = (coords, None, None, 0, None, None, None, None, None, f'kdtree_sky' if storekdtree else False, -1)
    else:
        if njobs>1:
            args = ((None, coords1_chunks[i], coords2_chunks[i], idxshift_list1[cidx1], idxshift_list1[cidx1], None, overidx_chunks1[job_num], overidx_chunks2[job_num] , f'kdtree_sky_{cidx1}_{cidx2}' if storekdtree else False, job_num) for job_num, (cidx1, cidx2) in enumerate(cross_idx))
        else:
            pass # ... add here


    
    if njobs>1:
        # with MPIPoolExecutor(max_workers=njobs) as pool:
        with multiprocessing.Pool(processes=njobs) as pool:
            # it is very crucial that Pool keeps the original order of data passed to starmap
            res = pool.starmap(make_partial_group_ids, args)

        with BusyPal('Concatenating the results from different processes', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}'):
            refidx = np.concatenate(refidx_chunks)
            if use_linked_mask:
                group_ids_chunks   = [item[0] for item in res]
                linked_mask_chunks = [item[1] for item in res]
                group_ids = np.concatenate(group_ids_chunks)
                linked_mask = np.concatenate(linked_mask_chunks)
                assert len(group_ids)==len(linked_mask)==len(refidx)
            else:
                group_ids = np.concatenate(res, axis=0)
                assert len(group_ids)==len(refidx)
                linked_mask = None

            del res, refidx_chunks
    else:
        # - serial
        group_ids = make_partial_group_ids(*args)
 

    # - merge the duplicates and find the underlying network shared between them!
    if njobs>1:
        # - stitch fragmented group ids from different patches/mosaics
        # # weed out first IF you don't have that many groups - many isolated (usually removes only 20% though, not worth it performance-wise)
        # if weedout_mask is None:
        #     # narrow things down a bit for faster grah computation
        #     cr, cg = Counter(refidx), Counter(group_ids)
        #     len_data = len(refidx)
        #     weedout_mask = [j for j in range(len_data) if cr[refidx[j]]>1 or (cg[group_ids[j]]>1)]
        #     print('=== len_data, len(weedout_mask) ', len_data, len(weedout_mask))

        if linked_mask is not None:
            with BusyPal('Stitch fragmented group ids from different mosaic sets', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}'):
                group_ids[linked_mask] = update_labels(refidx[linked_mask], group_ids[linked_mask])  #stitch_group_ids(refidx[linked_mask], group_ids[linked_mask], graph_lib=graph_lib, verbose=False, num_threads=-1) #update_labels(refidx[linked_mask], group_ids[linked_mask]) 
                # group_ids[linked_mask] = linked_group_ids
        else:
            with BusyPal('Stitch fragmented group ids from different mosaics - no linked_mask', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}'):
                group_ids = update_labels(refidx, group_ids) #stitch_group_ids(refidx, group_ids, graph_lib=graph_lib, verbose=False, num_threads=-1)
                # group_ids[weedout_mask] = update_labels(refidx[weedout_mask], group_ids[weedout_mask])

        # - put the two arrays in a dataframe
        df = pd.DataFrame({'idx': refidx, 'group_id': group_ids})


        with BusyPal('Rearrange indices to be the same as original input', style={'id':6,'color':'sandy_brown'}, fmt='{spinner} {message}'):
            # - we should drop duplicate idx records from the df
            df.drop_duplicates(subset='idx', keep='first', inplace=True)
            df.sort_values(by='idx', inplace=True)

        assert len(df)==num_objects

        group_ids = df['group_id']

    if reassign_group_indices:
        group_ids = np.unique(group_ids, return_inverse=True)[1]
    
    return group_ids
