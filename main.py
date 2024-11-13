import argparse
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from utils. common_utils import create_subfolders
from utils.game_utils import _is_reach_convergence,  _get_most_recent_generation, \
              _get_pr_of_strategy_update, _get_payoff_of_node, _get_pr_of_strategy_replacement

from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_, __N_GENERATIONS_, __EVENTS_
from configs.dilemmas import GAME_DICT
from configs.dilemmas import R, P
from configs.device_configs import device

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

import time

def __play_game_for_one_generation(node_attrs,adj_matrix,g,beta_e, beta_a, pr_strategy, lr, payoff_matrix):
    number_of_nodes = node_attrs.size(0)
    nodes = list(range(number_of_nodes))
    events = np.random.choice(__EVENTS_, size=number_of_nodes, p=[pr_strategy,1-pr_strategy])
    np.random.shuffle(nodes)
    for a, event in zip(nodes,events):
        if event == "strategy":
            st = time.time()
            # ngh_a =  (adj_matrix[a] == 1).nonzero().squeeze()
            ngh_a = list(g.neighbors(a))
            # print("ngh a: ", time.time()-st)
            # b = np.random.choice(ngh_a.cpu(), size=1)[0]
            b = np.random.choice(ngh_a, size=1)[0]
            str_b = node_attrs[b]

            st = time.time()
            # ngh_b = (adj_matrix[b] == 1).nonzero().squeeze()
            ngh_b = list(g.neighbors(b))
            # print("ngh b: ", time.time()-st)
            st = time.time()
            pi_a, pi_b = _get_payoff_of_node(a, ngh_a, node_attrs, payoff_matrix), _get_payoff_of_node(b, ngh_b, node_attrs, payoff_matrix)
            # print("payoff: ", time.time()-st)
            pr_strategy_replace = _get_pr_of_strategy_replacement(pi_a, pi_b, beta_e)
            
            st = time.time()
            is_replace = np.random.choice([True,False], size=1, p=[pr_strategy_replace,1-pr_strategy_replace])[0]
            # print("is replace: ", time.time()-st)
            if is_replace:   node_attrs[a] =  str_b   # strategy of a is replaced by B's strategy
        else:
            pass
            # print("## todo")

    
    return node_attrs, adj_matrix

def __play_game_for_g_generations(N, z, beta_e, beta_a, W, T, S, game, lr, simulation):
    node_attrs, adj_matrix, graph, gen_no = _get_most_recent_generation(N, z, beta_e, beta_a,  W, T, S, game, lr, simulation)
    pr_strategy = _get_pr_of_strategy_update(W)

    if _is_reach_convergence(node_attrs):
        print("Convergence was reached at gen_no: {}".format(gen_no))
        return
    
    # create payoff matrix
    payoff_matrix = torch.zeros((2,2))
    payoff_matrix[0][0] = P
    payoff_matrix[0][1] = T
    payoff_matrix[1][0] = S
    payoff_matrix[1][1] = R
    payoff_matrix = payoff_matrix.to(device)

    gen_generator =  tqdm(range(gen_no+1, __N_GENERATIONS_), desc='Running Generations')
    for g in gen_generator:
        node_attrs, adj_matrix = __play_game_for_one_generation(node_attrs, adj_matrix, graph,  beta_e, beta_a, pr_strategy, lr, payoff_matrix)
        if _is_reach_convergence(node_attrs):
            logging.info("Convergence is reached")
            print("##TODO more steps")
            break
        


def __play_game_for_n_simulations(N, z, beta_e, beta_a, W, T, S, game, lr):
    print("Running {} simulations for graph of N: {}, z:{}, game:{}, T: {}, S:{}".format(__N_INDEPENDENT_SIMULATIONS_,N,z,game,T,S))
    for simulation in range(__N_INDEPENDENT_SIMULATIONS_):
        logging.info("Running Simulation no : {}".format(simulation))
        __play_game_for_g_generations(N, z, beta_e, beta_a, W, T, S, game, lr, simulation)

def __play__game(N, z, beta_e, beta_a, game, W, lr):
    T_range, S_range = GAME_DICT[game]["T"], GAME_DICT[game]["S"]

    # for T in T_range:
    #     for S in S_range:
    T = 2
    S = -1
    __play_game_for_n_simulations(N, z, beta_e, beta_a, W, T, S, game, lr)
            # break
            
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", help="Time scale Ratio", type=float, default=0.0)
    parser.add_argument("--N", help="Number of Nodes", type=int, default=1000)
    parser.add_argument("--z", help="Average Connectivity",  type=int, default=30)
    parser.add_argument("--beta_e", help="Strategy Update", type=float, default=0.005)
    parser.add_argument("--beta_a", help="Adjustment of Ties", type=float, default=0.005)
    parser.add_argument("--game", help="Type of Social Dilemma (sg, sh, pd)", type=str)
    parser.add_argument("--lr", help="Possible types of link recommenders - rewiring", type=str, default="rewiring")


    args = parser.parse_args()
    __play__game(args.N, args.z, args.beta_e, args.beta_a, args.game, args.W, args.lr)