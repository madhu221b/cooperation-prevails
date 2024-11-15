import argparse
import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from joblib import Parallel, delayed
import multiprocessing
from utils. common_utils import create_subfolders, write_csv, read_csv
from utils.game_utils import _is_reach_convergence,  _get_init_graph, \
              _get_pr_of_strategy_update, _get_payoff_of_node, _get_pr_of_strategy_replacement, \
            _get_pr_of_adjust_ties, do_rewiring, _get_payoff_matrix

from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_, __N_GENERATIONS_, __EVENTS_, __HEADER_
from configs.dilemmas import GAME_DICT
from configs.dilemmas import R, P
from configs.device_configs import device

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

import time

def __play_game_for_g_generations(N, z, beta_e, beta_a, pr_W, T, S, game, lr, simulation,payoff_matrix):   
    node_attrs, adj_matrix = _get_init_graph(N, z, simulation)
    
    # logging.info("Running Simulation no : {}".format(simulation))
    number_of_nodes = node_attrs.size(0)
    nodes = list(range(number_of_nodes))
    is_convergence = False
    for g in range(__N_GENERATIONS_):  
        np.random.shuffle(nodes)
        events = np.random.choice(__EVENTS_, size=number_of_nodes, p=[pr_W,1-pr_W])
        
        for a, event in zip(nodes,events):

            is_conv, frac_c = _is_reach_convergence(node_attrs)
            if is_conv:
                # write_csv(filename, content=[simulation,frac_c,g])
                # is_convergence = True
                # break
                return [simulation,frac_c,g]
         
            ngh_a =  (adj_matrix[a] == 1).nonzero().squeeze(1).tolist()
            b = np.random.choice(ngh_a, size=1)[0]
            str_a, str_b = node_attrs[a], node_attrs[b]
            ngh_b = (adj_matrix[b] == 1).nonzero().squeeze(1).tolist()
            if event == "strategy":
                pi_a, pi_b = _get_payoff_of_node(a, ngh_a, node_attrs, payoff_matrix), _get_payoff_of_node(b, ngh_b, node_attrs, payoff_matrix)
                pr_strategy_replace = _get_pr_of_strategy_replacement(pi_a, pi_b, beta_e)
                is_replace = np.random.choice([True,False], size=1, p=[pr_strategy_replace,1-pr_strategy_replace])[0]
                if is_replace:   node_attrs[a] =  str_b   # strategy of a is replaced by B's strategy
            elif event == "structural":
                if lr == "rewiring":
                    adj_matrix = do_rewiring(a, b, ngh_a, ngh_b, str_a, str_b, node_attrs, adj_matrix, payoff_matrix, beta_a)
            else:
                logging.error("Received an event other than the two discussed")
        
        if is_convergence: break
    time.sleep(0.001)

def __play_game_for_n_simulations(N, z, beta_e, beta_a, W, T, S, game, lr, payoff_matrix):
    print("Running {} simulations for graph of N: {}, z:{}, game:{}, T: {}, S:{}".format(__N_INDEPENDENT_SIMULATIONS_,N,z,game,T,S))
    pr_W = _get_pr_of_strategy_update(W)
    
    folder_name = "./data/lr_{}/game_{}/W_{}_N_{}_z_{}_betae_{}_betaa_{}/".format(lr,game,W,N,z,beta_e,beta_a)
    create_subfolders(folder_name)
    # for simulation in sim_generator:
    #     logging.info("Running Simulation no : {}".format(simulation))
    #     __play_game_for_g_generations(N, z, beta_e, beta_a, W, T, S, game, lr, simulation,folder_name, payoff_matrix)
    filename = os.path.join(folder_name,"_T_{}_S_{}_run.csv".format(T,S))
    num_cores = 1
    if not os.path.exists(filename): 
           write_csv(filename, __HEADER_, mode="w")
           done_simulations = list()
    else:
        df = read_csv(filename)
        done_simulations = list(df["run_no"])

    remainder_simulations = list(set(range(__N_INDEPENDENT_SIMULATIONS_)) - set(done_simulations)) 
    if len(remainder_simulations) == 0: return
    sim_generator =  tqdm(remainder_simulations, desc='Running Simulations') 
    results = [Parallel(n_jobs=num_cores)(delayed(__play_game_for_g_generations)(N, z, beta_e, beta_a, pr_W, T, S, game, lr, simulation,payoff_matrix) for simulation in sim_generator)]
    print(results, file=open("trial.txt", "a"))

def __play__game(N, z, beta_e, beta_a, game, W, lr):
    T_range, S_range = GAME_DICT[game]["T"], GAME_DICT[game]["S"]
    
    st = time.time()
    for T in T_range:
        for S in S_range:
            payoff_matrix = _get_payoff_matrix(T,S)
            __play_game_for_n_simulations(N, z, beta_e, beta_a, W, T, S, game, lr, payoff_matrix)
      
   

    print("time taken for n simulations: ", time.time()-st)
            
        



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
    print("Simulations running on device: ", device)
    __play__game(args.N, args.z, args.beta_e, args.beta_a, args.game, args.W, args.lr)