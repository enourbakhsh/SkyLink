import igraph
import networkit as nk
import networkx as nx
from itertools import chain
import datetime


class GraphDataStructure(object):
    """The class object for graph analysis using three independent libraries.

    Parameters
    ----------
    graph_lib : str, default: 'igraph'
        The library used for the graph analysis. Options are `networkit`,
        `igraph`, and `networkx`.
    num_threads : int, optional, default: None
        Number of OpenMP threads (only applies to the `networkit` library)

    Raises
    ------
    ValueError
        If the user sets a value for `num_threads` without using `networkit` as
        the graph library of choice.

    Examples
    --------
    To find connected components of a graph using clustering:

    >>> gds = GraphDataStructure('networkit', num_threads=4)
    >>> edges = [(1,3), (2,5), (3,4), (4,1)]
    >>> graph = gds.build_graph_from_edges(edges)
    >>> gds.cluster(graph)
    It took 0:00:00 hms to identify the numberof nodes, i.e. `nnodes`=6.
    `networkit` graph built in 0:00:00 hms.
    `networkit` graph clustered in 0:00:00 hms.
    [[0], [1, 3, 4], [2, 5]]
    """

    def __init__(self, graph_lib, num_threads=None):
        self.graph_lib = graph_lib  # 'igraph', 'networkx', 'networkit'
        if graph_lib == "networkit":
            self.num_threads = (
                num_threads
                if num_threads != -1 and num_threads is not None
                else nk.getMaxNumberOfThreads()
            )  # OpenMP threads only for `networkit`.
            nk.setNumberOfThreads(self.num_threads)
        elif num_threads is not None:
            raise ValueError('`num_threads` is only used for the `networkit` library.')

    def build_graph_from_edges(self, edges, verbose=True, **kwargs):
        """Uses the input edges to construct a graph.

        Parameters
        ----------
        edges : int array, int list
            The input graph edges as a list or array of two-element lists or
            tuples.
            Example: [(1,9), (2,3), (4,7), ...]
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.
        **kwargs : dict
            Additional keyword arguments for the `self.build_{graph_lib}`
            function.

        Returns
        -------
        graph : `self.graph_lib` graph object
            The output graph made from the input edges.
        """

        t0 = datetime.datetime.now()
        graph = getattr(self, f"build_{self.graph_lib}")(
            edges, verbose=verbose, **kwargs
        )
        elapsed_seconds = round((datetime.datetime.now()-t0).seconds)
        if verbose:
            print(
                f"`{self.graph_lib}` graph built in {str(datetime.timedelta(seconds=elapsed_seconds))} hms."
            )

        return graph

    def merge(self, graph_list, verbose=True):
        """Merges multiple graphs into one graph.

        Parameters
        ----------
        graph_list : `self.graph_lib` graph list or `self.graph_lib` graph
            array
            The list of input graphs.
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.

        Returns
        -------
        graph : `self.graph_lib` graph object
            The output graph made by merging the input graphs in the
            `graph_list`.
        """

        t0 = datetime.datetime.now()
        graph_merged = getattr(self, f"merge_{self.graph_lib}")(graph_list)
        elapsed_seconds = round((datetime.datetime.now()-t0).seconds)
        if verbose:
            print(
                f"{self.graph_lib} graphs merged in {str(datetime.timedelta(seconds=elapsed_seconds))} hms."
            )

        return graph_merged

    def cluster(self, graph, verbose=True):
        """Finds clusters in a graph. Each cluster is a list of nodes.

        Parameters
        ----------
        graph : `self.graph_lib` graph object
            The input graph.
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.

        Returns
        -------
        clusters : `cluster` object of `self.graph_lib`
            Clusters found using a connected component analysis.
        """

        if isinstance(graph, (list, tuple)):
            # Merge them first (graphs don't have to be disjoint).
            graph = self.merge(graph, verbose=verbose)
        t0 = datetime.datetime.now()
        clusters = getattr(self, f"cluster_{self.graph_lib}")(graph)
        elapsed_seconds = round((datetime.datetime.now()-t0).seconds)
        if verbose:
            print(
                f"`{self.graph_lib}` graph clustered in"
                f"{str(datetime.timedelta(seconds=elapsed_seconds))} hms."
            )

        return clusters

    @staticmethod
    def build_networkx(edges, verbose=True, **kwargs):
        """Builds a `networkx` graph from the input edges.

        Parameters
        ----------
        edges : array or list
            The input graph edges as a list or array of two-element lists or
            tuples.
            Example: [(1,9), (2,3), (4,7), ...]
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.
        **kwargs : dict
            Additional keyword arguments for the `networkx.Graph`
            function.

        Returns
        -------
        G : `networkx` graph object
            The output graph made from the input edges.
        """

        edges = list(edges)
        G = nx.Graph(edges)  # **kwargs

        return G

    @staticmethod
    def build_igraph(edges, verbose=True, **kwargs):
        """Builds a `igraph` graph from the input edges.

        edges : array or list
            The input graph edges as a list or array of two-element lists or
            tuples.
            Example: [(1,9), (2,3), (4,7), ...]
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.
        **kwargs : dict
            Additional keyword arguments for the `igraph.Graph`
            function.

        Returns
        -------
        G : `igraph` graph object
            The output graph made from the input edges.
        """

        return igraph.Graph(edges)

    @staticmethod
    def build_networkit(edges, verbose=True, nnodes=None):
        """Builds a `networkit` graph from the input edges.

        edges : array or list
            The input graph edges as a list or array of two-element lists or
            tuples.
            Example: [(1,9), (2,3), (4,7), ...]
        verbose : bool or int, optional, default: True
            If True or 1, it produces lots of logging output.
        **kwargs : dict
            Additional keyword arguments for the `networkit.Graph`
            function.

        Returns
        -------
        G : `networkit` graph object
            The output graph made from the input edges.
        """

        if nnodes is None:
            t0 = datetime.datetime.now()
            unraveled = list(
                chain.from_iterable(edges)
            )  # TODO: np.concat is faster!
            if not isinstance(unraveled[0], str) and not isinstance(unraveled[1], str):
                nnodes = max(unraveled) + 1  # just enough nodes to construct the graph
            else:
                nnodes = len(unraveled)  # length of ('a',5), ('f',2), ...
            if verbose:
                # Important: do not rely on this to cover all the isolated
                # coords (as isolated nodes in the graph).
                # That's why we used `np.arange` later to fill in the possible
                # missing bits.
                elapsed_seconds = round((datetime.datetime.now()-t0).seconds)
                print(
                    f"It took {str(datetime.timedelta(seconds=elapsed_seconds))} hms to identify the number"
                    f"of nodes, i.e. `nnodes`={nnodes}."
                )
            del unraveled

        G = nk.Graph(nnodes, directed=False)

        for edge in edges:
            G.addEdge(*edge)

        return G

    @staticmethod
    def merge_networkx(graph_list):
        """Merges multiple `networkx` graphs into one graph.

        Parameters
        ----------
        graph_list : `networkx` graph list or `networkx` graph array
            The list of input graphs.

        Returns
        -------
        graph_merged : `networkx` graph object
            The output graph made by merging the input graphs in the
            `graph_list`.
        """

        # The simple union of the node sets and edge sets. The node sets of G
        # and H do not need to be disjoint.
        graph_merged = graph_list[0]

        for graph in graph_list[1:]:
            graph_merged = nx.compose(graph_merged, graph)
            # This `merged graph has all nodes and edges of both graphs (the
            # `compose` arguments), including the attributes. Where the
            # attributes conflict, it uses the attributes of the second
            # argument/graph.

        return graph_merged

    @staticmethod
    def merge_igraph(graph_list):
        """Merges multiple `igraph` graphs into one graph.

        Parameters
        ----------
        graph_list : `igraph` graph list or `igraph` graph array
            The list of input graphs.

        Returns
        -------
        graph_merged : `igraph` graph object
            The output graph made by merging the input graphs in the
            `graph_list`.
        """

        # The simple union of the node sets and edge sets. The node sets of G
        # and H do not need to be disjoint.
        graph_merged = graph_list[0].union(graph_list[1:])

        return graph_merged

    @staticmethod
    def merge_networkit(graph_list):
        """Merges multiple `networkit` graphs into one graph.

        Parameters
        ----------
        graph_list : `networkit` graph list or `networkit` graph array
            The list of input graphs.

        Returns
        -------
        graph_merged : `networkx` graph object
            The output graph made by merging the input graphs in the
            `graph_list`.
        """

        # Modifies this graph to be the union of it and another graph. Nodes
        # with the same ids are identified with each other.
        graph_merged = graph_list[0]

        for graph in graph_list[1:]:
            nk.graphtools.merge(graph_merged, graph)

        return graph_merged

    @staticmethod
    def cluster_networkx(G):
        """Finds clusters in a `networkx` graph. Each cluster is a list of
        nodes.

        Parameters
        ----------
        G : `networkx` graph object
            The input graph.

        Returns
        -------
        list
            Clusters found using a connected component analysis.
        """

        cc = nx.connected_components(G)

        return list(map(list, cc))

    @staticmethod
    def cluster_igraph(G):
        """Finds clusters in a `igraph` graph. Each cluster is a list of
        nodes.

        Parameters
        ----------
        G : `igraph` graph object
            The input graph.

        Returns
        -------
        cc : `cluster` object of `igraph`
            Clusters found using a connected component analysis.
        """

        # `mode=STRONG` by default
        # In case of an undirected graph (like ours), a weakly connected
        # component is also a strongly connected component.
        cc = G.clusters()

        return cc

    @staticmethod
    def cluster_networkit(G):
        """Finds clusters in a `networkit` graph. Each cluster is a list of
        nodes.

        Parameters
        ----------
        G : `networkit` graph object
            The input graph.

        Returns
        -------
        cc : `cluster` object of `networkit`
            Clusters found using a connected component analysis.
        """

        cc_obj = nk.components.ConnectedComponents(G)
        cc_obj.run()
        cc = cc_obj.getComponents()

        return cc
