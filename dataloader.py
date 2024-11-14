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

from utils.common_utils import set_seed, read_pickle, save_pickle


def _get_degree(g):
    "Get degree of the graph"
    return sum([d for (n, d) in nx.degree(g)]) / float(g.number_of_nodes())

def _set_strategy_randomly(g, per_c=0.5):
    """
    Setting attribute 1 for cooperators and 0 for defectors
    """
    N = g.number_of_nodes()
    prop_c = int(N*per_c)
    prop_d = N - prop_c
    array = [1] * (prop_c) + [0] * (prop_d)
    np.random.shuffle(array)
    print("Setting {} cooperators and {} defectors for {} nodes randomly".format(prop_c, prop_d, N))
    idxs = range(N)
    node_dict = dict(zip(idxs, array))
    nx.set_node_attributes(g, node_dict, name="strategy")
    return g
    

def _generate_graph(N, z, seed):
    """
    N : number of nodes
    z : degree
    
    E = Nz/2 (no of edges)
    Process : 
    while continue till fE edges are successfully rewired: 
        (1) choose—randomly and independently—two different edges which have not been used 
        (2) Swap the ends of the two edges if no duplicate connections arise
    
    
    """
    path = "./data/init_graphs/_N_{}_z_{}_seed_{}.gpickle".format(N,z,seed)
    g_obj = read_pickle(path)
    if g_obj is not None: 
        g = g_obj
    else:
        set_seed(seed)
        g = nx.watts_strogatz_graph(N, z, p=1, seed=seed)    
        # Assign 50-50% Cooperators and Defectors
        g = _set_strategy_randomly(g, per_c=0.5)
        print("Saving graph at {}".format(path))
        save_pickle(g, path)

    return g

