import networkx as nx
import os
from utils. common_utils import create_subfolders
from dataloader import _generate_graph

def _is_reach_convergence(graph):
    node_dict = nx.get_node_attributes(graph, "strategy")
    return sum(list(node_dict.values())) == graph.number_of_nodes()

def _get_most_recent_generation(N, z, beta, W, T, S, game, lr, simulation):

    folder_path = "./data/lr_{}/game_{}_W_{}_N_{}_z_{}_beta_{}/_T_{}_S_{}/_run_{}/".format(lr,game,W,N,z,beta,T,S,simulation)
    create_subfolders(folder_path)

    files = os.listdir(folder_path)
    if len(files) == 0:
        print("Fetching Initial Graph")
        g = _generate_graph(N, z)
        gen_no = -1
    else:
        print("## TO DO")
    
    return g, gen_no

def _get_pr_of_strategy_update(W):
    return 1/(1+W)