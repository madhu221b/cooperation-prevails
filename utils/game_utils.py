import networkx as nx
import os
import numpy as np
import torch
from utils.common_utils import create_subfolders
from configs.dilemmas import R, P
from configs.device_configs import device
from dataloader import _generate_graph

def _is_reach_convergence(node_attrs):
    print("~~", torch.sum(node_attrs))
    return torch.sum(node_attrs) == node_attrs.size(0) or torch.sum(node_attrs) == 0

def _get_most_recent_generation(N, z, beta_e, beta_a, W, T, S, game, lr, simulation):

    folder_path = "./data/lr_{}/game_{}_W_{}_N_{}_z_{}_betae_{}_betaa_{}/_T_{}_S_{}/_run_{}/".format(lr,game,W,N,z,beta_e,beta_a,T,S,simulation)
    create_subfolders(folder_path)

    files = os.listdir(folder_path)
    if len(files) == 0:
        print("Fetching Initial Graph")
        g = _generate_graph(N, z)
        gen_no = -1
    else:
        print("## TO DO")
    
    # get tensor of node attributes
    node_attrs = nx.get_node_attributes(g, "strategy")
    node_tensors = torch.zeros(g.number_of_nodes())
    for node, strategy in node_attrs.items():
        node_tensors[node] = strategy

    # get adjacency matrix
    adj_matrix =  nx.to_numpy_array(g, nodelist=list(range(g.number_of_nodes()))).astype(np.float32)
    adj_matrix = torch.tensor(adj_matrix)

    node_tensors = node_tensors.to(device)
    adj_matrix = adj_matrix.to(device)
    return node_tensors, adj_matrix, g, gen_no

def _get_pr_of_strategy_update(W):
    return 1/(1+W)

def _get_payoff_of_node(a, ngh_a, node_attrs, payoff_matrix): 
    node_id = node_attrs[a]
    node_id_nghs = node_attrs[ngh_a]
    vals = payoff_matrix[node_id.int(), node_id_nghs.int()]
    return torch.sum(vals)

def _get_pr_of_strategy_replacement(pi_a, pi_b, beta):
    pi_a, pi_b = pi_a.cpu().numpy(), pi_b.cpu().numpy()
    power = -beta*(pi_b-pi_a)
    return 1/(1+np.exp(power))