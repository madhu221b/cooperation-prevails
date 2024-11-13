"""
We generate the initial graphs required in this script.
Input Graph -   Homogeneous random graph (HoSW) -  in which all nodes have the same number of edges
                Randomly linked to arbitrary nodes

Process to generate such graph mentioned in paper - 
Epidemic spreading and cooperation dynamics on homogeneous small-world networks

Undirected Regular Graph (input) -> HoSW (output)
"""

import numpy as np
import networkx as nx

from utils.common_utils import set_seed


def _get_degree(g):
    "Get degree of the graph"
    return sum([d for (n, d) in nx.degree(g)]) / float(g.number_of_nodes())

def _generate_graph(N, z):
    """
    N : number of nodes
    z : degree
    
    E = Nz/2 (no of edges)
    Process : 
    while continue till fE edges are successfully rewired: 
        (1) choose—randomly and independently—two different edges which have not been used 
        (2) Swap the ends of the two edges if no duplicate connections arise
    
    
    """
    seed = 42
    f = 1 # Setting this intutively
    set_seed(seed)

    g = nx.random_regular_graph(z, N, seed)
    init_edges = set(list(g.edges()))
    print("Random Regular graph has N : {}, E: {}, avg degree: {}".format(g.number_of_nodes(), g.number_of_edges(), _get_degree(g)))

    edges = list(g.edges())
    quota = f*len(edges)

    while quota > 0:
        
        idxs = np.random.choice(range(len(edges)), 2)
        a, b = edges[idxs[0]]
        e, f = edges[idxs[1]]

        if g.has_edge(a,f) or g.has_edge(b, e): # checking if connections for possible swap already exist
           continue
        
        g.remove_edges_from([(a,b), (e,f)])
        g.add_edges_from([(a,f), (e,b)])

        edge_set = set(edges)
        edge_set.remove((a,b))
        edge_set.remove((e,f))
        edges = list(edge_set)

        quota -= 2 # 2 edges are rewired
    
    print("HoSW graph has N : {}, E: {}, avg degree: {}".format(g.number_of_nodes(), g.number_of_edges(),_get_degree(g)))
    final_edges = set(list(g.edges()))
    assert init_edges != final_edges, "Regular Random Graph and HoSW have the same edges"
    return g


g = _generate_graph(1000,40)