import networkx as nx
import os
import numpy as np
import torch
from utils.common_utils import create_subfolders
from configs.dilemmas import R, P
from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_
from configs.device_configs import device
from dataloader import _generate_graph

def _is_reach_convergence(node_attrs):
    """
    Input: the strategy of nodes
    Output: whether convergence is reached (all C or all D), fraction of cooperators
    """
    cooperators = torch.sum(node_attrs)
    return (cooperators == node_attrs.size(0) or cooperators == 0, float(cooperators/node_attrs.size(0)))

def _get_init_graph(N, z, seed):
    g = _generate_graph(N, z, seed)
    # get tensor of node attributes
    node_attrs = nx.get_node_attributes(g, "strategy")
    node_tensors = torch.zeros(g.number_of_nodes(), dtype=int)
    for node, strategy in node_attrs.items():
        node_tensors[node] = strategy

    # get adjacency matrix
    adj_matrix =  nx.to_numpy_array(g, nodelist=list(range(g.number_of_nodes()))).astype(np.float32)
    adj_matrix = torch.tensor(adj_matrix)

    node_tensors = node_tensors.to(device)
    adj_matrix = adj_matrix.to(device)
    return node_tensors, adj_matrix

def _get_init_graphs(N, z, seeds):

    node_attr_big = torch.zeros((__N_INDEPENDENT_SIMULATIONS_,N), dtype=int)
    adj_matrix_big = torch.zeros((__N_INDEPENDENT_SIMULATIONS_,N,N))


    for i, seed in enumerate(seeds):
        g = _generate_graph(N, z, seed)
        # get tensor of node attributes
        node_attrs = nx.get_node_attributes(g, "strategy")
        node_tensors = torch.zeros(N)
        for node, strategy in node_attrs.items():
             node_tensors[node] = strategy

        # get adjacency matrix
        adj_matrix =  nx.to_numpy_array(g, nodelist=list(range(g.number_of_nodes()))).astype(np.float32)
        adj_matrix = torch.tensor(adj_matrix)

        node_attr_big[seed] = node_tensors
        adj_matrix_big[seed] = adj_matrix
      
    node_attr_big = node_attr_big.to(device)
    adj_matrix_big = adj_matrix_big.to(device)
    return node_attr_big, adj_matrix_big

def _get_pr_of_strategy_update(W):
    return 1/(1+W)

def _get_payoff_matrix(T,S):
    payoff_matrix = torch.zeros((2,2))
    payoff_matrix[0][0] = P
    payoff_matrix[0][1] = T
    payoff_matrix[1][0] = S
    payoff_matrix[1][1] = R
    payoff_matrix = payoff_matrix.to(device)
    return payoff_matrix

def _get_payoff_of_node(a, ngh_a, node_attrs, payoff_matrix): 
    node_id = node_attrs[a]
    node_id_nghs = node_attrs[ngh_a]
    vals = payoff_matrix[node_id.int(), node_id_nghs.int()]
    return torch.sum(vals)

def _get_payoff_of_nodes(nodes,nghs,node_attrs, payoff_matrix): 
    payoffs = list()
    for node, ngh in zip(nodes, nghs):
        node_id = node_attrs[node]
        node_id_nghs = node_attrs[ngh]
        vals = payoff_matrix[node_id.int(), node_id_nghs.int()]
        payoffs.append(torch.sum(vals))
    return payoffs

# def _get_payoff_of_nodes_for_all_sims(ngh_dict, node_attrs, payoff_matrix, sims_set):
#     payoff_tensor = torch.zeros((sims_set.size(0), 3)) # sim_no, payoff_a, payoff_b

#     for i, sim_no in enumerate(sims_set):
#         sim_no = int(sim_no)
#         a, b = ngh_dict[sim_no]["a"], ngh_dict[sim_no]["b"]
#         node_id_a, node_id_b = node_attrs[sim_no][a], node_attrs[sim_no][b]
#         ngh_a, ngh_b = ngh_dict[sim_no]["ngh_a"], ngh_dict[sim_no]["ngh_b"]
#         node_id_nghs_a, node_id_nghs_b = node_attrs[sim_no][ngh_a], node_attrs[sim_no][ngh_b]
#         vals_a, vals_b = payoff_matrix[node_id_a, node_id_nghs_a], payoff_matrix[node_id_b, node_id_nghs_b]
#         payoff_tensor[i] = torch.tensor([sim_no, torch.sum(vals_a), torch.sum(vals_b)])

#     return payoff_tensor

def _get_payoff_of_nodes_for_all_sims(all_nodes, node_attrs, adj_matrix_all, payoff_matrix, sims_set, sim_ab_tensor):
    payoff_tensor = torch.zeros((sims_set.size(0), 3)) # sim_no, payoff_a, payoff_b

    sim_ab_tensor_subset = sim_ab_tensor[torch.isin(sim_ab_tensor[:, 0], sims_set)]
    sim_nos = sim_ab_tensor_subset[:, 0]
    a_s = sim_ab_tensor_subset[:, 1]
    node_id_a_s = node_attrs[sim_nos, a_s]
    b_s = sim_ab_tensor_subset[: , 2]
    node_id_b_s = node_attrs[sim_nos, b_s]
    
    rep = all_nodes.repeat(sim_nos.size(0),1)
    node_id_all = node_attrs[sim_nos[:, None], rep]
    
    
    adj_matrix_subset = adj_matrix_all[sim_nos, a_s]
    pf = payoff_matrix[node_id_a_s[:, None], node_id_all]
    pf_needed = pf * adj_matrix_subset
    pf_sum_a = torch.sum(pf_needed, dim=-1)

    adj_matrix_subset = adj_matrix_all[sim_nos, b_s]
    pf = payoff_matrix[node_id_b_s[:, None], node_id_all]
    pf_needed = pf * adj_matrix_subset
    pf_sum_b = torch.sum(pf_needed, dim=-1)


    payoff_tensor[:, 0] = sim_nos
    payoff_tensor[:, 1] = pf_sum_a
    payoff_tensor[:, 2] = pf_sum_b

    return payoff_tensor
        
# def _get_pr_of_strategy_replacement(pi_a, pi_b, beta):
#     pi_a, pi_b = pi_a.cpu().numpy(), pi_b.cpu().numpy()
#     power = -beta*(pi_b-pi_a)
#     return 1/(1+np.exp(power))

# def _get_pr_of_adjust_ties(pi_a, pi_b, beta):
#     pi_a, pi_b = pi_a.cpu().numpy(), pi_b.cpu().numpy()
#     power = -beta*(pi_a-pi_b)
#     return 1/(1+np.exp(power))

def _get_pr_of_strategy_replacement(pi_a, pi_b, beta):
    power = -beta*(pi_b-pi_a)
    return 1/(1+torch.exp(power))

def _get_pr_of_adjust_ties(pi_a, pi_b, beta):
    power = -beta*(pi_a-pi_b)
    return 1/(1+torch.exp(power))

def do_rewiring(a, b, ngh_a, ngh_b, str_a, str_b, node_attrs, adj_matrix, payoff_matrix, beta_a):
    if str_a and str_b: 
        # both a and b are C, both are satisfied, do nothing.
        pass
    elif not str_b:  # (a= c/d, b = d) a is not satisifed, rewiring for a 
            pi_a, pi_b = _get_payoff_of_node(a, ngh_a, node_attrs, payoff_matrix), _get_payoff_of_node(b, ngh_b, node_attrs, payoff_matrix)
            pr_adjust_ties = _get_pr_of_adjust_ties(pi_a, pi_b, beta_a)
            is_adjust = np.random.choice([True,False], size=1, p=[pr_adjust_ties,1-pr_adjust_ties])[0]
            if is_adjust:
                candidates = list(set(ngh_b) - set(ngh_a+[a])) # neighbours of b and not neighbors of a
                if len(candidates) != 0:   # "b and a have exclusively different neighbors"
                    c = np.random.choice(candidates, size=1)[0] # select a candidate to rewire
                    adj_matrix[a][b] = adj_matrix[b][a] = 0
                    adj_matrix[a][c] = adj_matrix[c][a] = 1
    else:
        pass

    return adj_matrix


def get_fracs_at_g(node_attrs_all, non_convergence_idxs, is_convergence_all):

    sum_c = torch.sum(node_attrs_all[non_convergence_idxs], dim=-1)
    frac_c = (sum_c/N)

    sim_frac = torch.cat((non_convergence_idxs[:, None], frac_c[:, None]), dim=-1)
    for sim, frac in sim_frac:
        idx = (is_convergence_all[:, 0] == sim).nonzero()
        is_convergence_all[idx, 1] += frac
    
    return is_convergence_all

