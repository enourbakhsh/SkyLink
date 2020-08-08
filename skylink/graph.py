import igraph
import networkit as nk
import networkx as nx
from itertools import chain
# import numpy as np
import datetime

class GraphDataStructure:
    def __init__(self, graph_lib, num_threads=None):
        self.graph_lib = graph_lib # 'igraph', 'igraph', 'networkit'
        if graph_lib=='networkit':
            self.num_threads = num_threads if num_threads!=-1 and num_threads is not None else nk.getMaxNumberOfThreads() # openmp threads only for networkit
            # - networkit parallelism
            #   set the maximum number of available openmp threads
            # print('$$$$$$$-->',self.num_threads,nk.getMaxNumberOfThreads(),nk.getCurrentNumberOfThreads())
            nk.setNumberOfThreads(self.num_threads)
            # print('$$***$$-->',self.num_threads,nk.getMaxNumberOfThreads(),nk.getCurrentNumberOfThreads())
        # elif num_threads is not None:
        #     raise ValueError('`num_threads` is only used for the `networkit` library.')

    def build_graph_from_edges(self, edges, verbose=True, **kwargs):
        t0 = datetime.datetime.now()
        graph = getattr(self, f'build_{self.graph_lib}')(edges, verbose=verbose, **kwargs)
        if verbose:
            print(f'{self.graph_lib} graph built in {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')
        return graph

    def merge(self, graph_list, verbose=True):
        t0 = datetime.datetime.now()
        graph_merged = getattr(self, f'merge_{self.graph_lib}')(graph_list)
        if verbose:
            print(f'{self.graph_lib} graphs merged in {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')
        return graph_merged

    def cluster(self, graph, verbose=True):
        if isinstance(graph, (list, tuple)):
            # - merge them first (graphs don't have to be disjoint)
            graph = self.merge(graph, verbose=verbose)
        t0 = datetime.datetime.now()
        clusters = getattr(self, f'cluster_{self.graph_lib}')(graph)
        if verbose:
            print(f'{self.graph_lib} graph clustered in {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms.')
        return clusters

    @staticmethod
    def build_networkx(edges, verbose=True, **kwargs):
        edges = list(edges)
        G=nx.Graph(edges)
        # G.add_nodes_from(edges) # ???
        # G.add_edges_from(edges)
        # for l in edges: #tqdm(edges, total=len(edges), desc='igraph'):
        #     #G.add_nodes_from(l)
        #     nx.add_path(G, l)
        return G

    @staticmethod
    def build_igraph(edges, verbose=True, **kwargs):
        return igraph.Graph(edges)

    @staticmethod
    def build_networkit(edges, verbose=True, nnodes=None):
        # print('nk.getCurrentNumberOfThreads(), max=',nk.getCurrentNumberOfThreads(), nk.getMaxNumberOfThreads())
        # # networkit parallelism
        # nk.setNumberOfThreads(self.num_threads) # set the maximum number of available openmp threads
        # nk.getMaxNumberOfThreads() # see maximum number of available threads
        # nk.getCurrentNumberOfThreads() # the number of threads currently executing


        # if nnodes is None:
        #print('chain.from_iterable(edges)',list(chain.from_iterable(edges)))
        # nnodes = np.max(list(chain.from_iterable(edges)))+1 #max(chain.from_iterable(edges))+1 # just enough nodes to construct the graph
        
        if nnodes is None:
            t0 = datetime.datetime.now()
            unraveled = list(chain.from_iterable(edges)) #--> !! np.concat is way faster!!!! change it
            # print('*****',unraveled[0])
            if not isinstance(unraveled[0], str) and not isinstance(unraveled[1], str):
                nnodes = max(unraveled)+1 # just enough nodes to construct the graph
            else:
                nnodes = len(unraveled) #?? ('a',5), ('f',2), ...
            if verbose:
                # important: do not rely on this to cover all the isolated coords (as isolated nodes in the graph)
                # that's why we used `np.arange` later to fill in the possible missing bits
                print(f'--nnodes={nnodes} took {str(datetime.timedelta(seconds=round((datetime.datetime.now()-t0).seconds)))} hms to find')
            del unraveled

        G = nk.Graph(nnodes, directed=False)


        for edge in edges:
            G.addEdge(*edge)
        return G

    @staticmethod
    def merge_networkx(graph_list):
        # - the simple union of the node sets and edge sets. The node sets of G and H do not need to be disjoint.
        graph_merged = graph_list[0]
        for graph in graph_list[1:]:
            graph_merged = nx.compose(graph_merged, graph)
            # `mega_graph` has all nodes and edges of both graphs (the compose arguments), including attributes
            # Where the attributes conflict, it uses the attributes of the second argument/graph.
        return graph_merged

    @staticmethod
    def merge_igraph(graph_list):
        # - the simple union of the node sets and edge sets. The node sets of G and H do not need to be disjoint.
        graph_merged = graph_list[0].union(graph_list[1:])
        return graph_merged

    @staticmethod
    def merge_networkit(graph_list):
        # Modifies this graph to be the union of it and another graph. Nodes with the same ids are identified with each other.
        graph_merged = graph_list[0]
        for graph in graph_list[1:]:
            nk.graphtools.merge(graph_merged, graph)
        return graph_merged

    @staticmethod
    def cluster_networkx(G):
        cc = nx.connected_components(G)
        return list(map(list, cc))

    @staticmethod
    def cluster_igraph(G):
        cc = G.clusters() # mode=STRONG by default # In case of an undirected graph (like ours), a weakly connected component is also a strongly connected component.
        return cc

    @staticmethod # forcefully make it a regular method so that num_threads in __init__ is set (user can't direcly call this?)
    def cluster_networkit(G):
        cc_obj = nk.components.ConnectedComponents(G)
        cc_obj.run()
        cc = cc_obj.getComponents()
        return cc
