import argparse
import os
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool
from utils. common_utils import create_subfolders, write_csv, read_csv
from utils.game_utils import _is_reach_convergence,  _get_init_graphs, \
              _get_pr_of_strategy_update, _get_payoff_of_node, _get_payoff_of_nodes, _get_payoff_of_nodes_for_all_sims, \
               _get_pr_of_strategy_replacement, \
            _get_pr_of_adjust_ties, do_rewiring, _get_payoff_matrix

from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_, __N_GENERATIONS_, __EVENTS_, __HEADER_
from configs.dilemmas import GAME_DICT, chunk_dict
from configs.dilemmas import R, P
from configs.device_configs import device

import logging
import threading
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

import time

N = 1000
nodes = torch.arange(N).to(device)
number_of_nodes = N

z = 30
beta_e = 0.005
beta_a = 0.005
W = 0.0
pr_W = _get_pr_of_strategy_update(W)
pr_Ws = torch.tensor([pr_W,1-pr_W],device=device) # __EVENTS_ = = ["strategy","structural"]
lr = "rewiring"

OFFSET = 1000


def __play_game_for_g_generations(T, S, game, simulations,filename,payoff_matrix):   
    n_simulations = len(simulations)
    print("T: {}, S:{} - Running  len Simulations  : {}".format(T,S, len(simulations)))
    
     # shape of node_attrs = [n_simulations, N] adj_matrix = [n_simulations,N,N]
    node_attrs_all, adj_matrix_all, mapping_id2sim = _get_init_graphs(N, z, simulations)
    is_convergence_all = torch.zeros(n_simulations, device=device) # we maintain for all simulations, whether we reached convergence here


    for g in range(__N_GENERATIONS_):  
        print("generation  g: ", g)
        if torch.all(is_convergence_all): break # all simulations reached convergence
         
        non_convergence_idxs = (is_convergence_all == 0).nonzero().squeeze(1)
        remainder_sims = non_convergence_idxs.size(0)
   
   
     
        ##Shuffling of Nodes
        shuffled_nodes = torch.zeros((remainder_sims, N), dtype=int) # n_sims X N
        for i in range(remainder_sims):                                                                                               # shuffle nodes for all simulations
            shuffled_nodes[i] = nodes[torch.randperm(N)]  # shuffle the nodes
       
        
        pr_Ws_tensor = pr_Ws.repeat(remainder_sims,1) # n_sims X 2
    
        events = torch.multinomial(pr_Ws_tensor,N,replacement=True)  # n_sims X N
        
        assert shuffled_nodes.size() == events.size()

       
        for i in range(N): # select one agent iteratively out of N agents
            a_for_all_sims = shuffled_nodes[:,i] # select source nodes for all simulations # sim
            events_at_i_for_all_sims = events[:,i]
            
            # 1. ngh_a = find neighbors of a, ngh_b = find neighbours of b
            ab_and_nghs = dict()
            replacement_tensor = torch.zeros((len(non_convergence_idxs),2),dtype=int, device=device) # sim_no, a, node_attr_b
            for i, (sim_no, a) in enumerate(zip(non_convergence_idxs,a_for_all_sims)):
                potential_bs = (adj_matrix_all[sim_no,a,:] == 1).nonzero()
                rand_int = torch.randint(0, potential_bs.size(0), (1,))[0]
                b = potential_bs[rand_int]
                ngh_bs =  (adj_matrix_all[sim_no,b,:] == 1).nonzero()
                ab_and_nghs[sim_no] = {"a":a, "b":b, "ngh_a":potential_bs, "ngh_b":ngh_bs}
                replacement_tensor[i] = torch.tensor([sim_no, node_attrs_all[sim_no][b]])
            
            
        
            # 2. do strategy events for nodes of all simulations
            sim_idxs_struct = (events_at_i_for_all_sims == 0).nonzero()
            struct_dict = {k:v for k, v in ab_and_nghs.items() if k in sim_idxs_struct}
            payoff_tensor = _get_payoff_of_nodes_for_all_sims(struct_dict, node_attrs_all, payoff_matrix) # sim_nos x 3
            pr_strategy_replace = _get_pr_of_strategy_replacement(payoff_tensor[:,1], payoff_tensor[:,2], beta_e) # sim_nos X 1
            prs_str = torch.vstack([pr_strategy_replace, 1-pr_strategy_replace]).T
            replace_idx = torch.multinomial(prs_str, 1) # sim_nos X 1
            sims_replace = sim_idxs_struct[replace_idx == 0] # replace idxs = 0 , pr_strategy_replace selected
            node_attrs_all[sims_replace, a] =  replacement_tensor[sims_replace, 1]

            
            


            # 3. do structural events for nodes of all simulations
            sim_idxs_struct = (events_at_i_for_all_sims == 1).nonzero()
            


        # 4. Check for convergence
        check_for_sims = (is_convergence_all == 0).nonzero()
        sum_c = torch.sum(node_attrs_all[check_for_sims], dim=-1)
        frac_c = sum_c/N
        # print(frac_c)
        defector_idxs = check_for_sims[(frac_c == 0.0).nonzero().squeeze(1)]
        cooperator_idxs = check_for_sims[(frac_c == 1.0).nonzero().squeeze(1)]
        converge_idxs = torch.cat((defector_idxs,cooperator_idxs),-1)
        # print(converge_idxs.size(), converge_idxs)
        if converge_idxs.size(0) != 0: 
            print("convergence reached for gen: g {}, convrge idex: {}".format(g,converge_idxs))
            is_convergence_all[converge_idxs] = 1.0




          

        

            

        # for a, event in None:

        #     is_conv, frac_c = _is_reach_convergence(node_attrs)
        #     if is_conv:
        #         logging.info("Convergence reached for T:{}, S:{}, sim no: {}".format(T,S,simulation))
        #         # return [simulation,frac_c,g]
        #         write_csv(filename, content=[simulation,frac_c,g])
        #         is_convergence = True
        #         break
         
        #     ngh_a =  (adj_matrix[a] == 1).nonzero()
        #     b = ngh_a[torch.randint(0, ngh_a.size(0), (1,))[0]]
        #     str_a, str_b = node_attrs[a], node_attrs[b]
        #     ngh_b = (adj_matrix[b] == 1).nonzero()
            
        #     if event == 0:
        #         payoffs = _get_payoff_of_nodes([a,b],[ngh_a,ngh_b],node_attrs,payoff_matrix)
        #         pi_a, pi_b = payoffs[0], payoffs[1]
        #         pr_strategy_replace = _get_pr_of_strategy_replacement(pi_a, pi_b, beta_e)
        #         replace_idx = torch.multinomial(torch.tensor([pr_strategy_replace,1-pr_strategy_replace]), 1)[0]
        #         if replace_idx == 0:   node_attrs[a] =  str_b   # strategy of a is replaced by B's strategy
        #     elif event == 1:
        #         if lr == "rewiring":
        #             adj_matrix = do_rewiring(a, b, ngh_a, ngh_b, str_a, str_b, node_attrs, adj_matrix, payoff_matrix, beta_a)
        #     else:
        #         logging.error("Received an event other than the two discussed")
        
        
        # if is_convergence:
        #     break
   

def __play_game_for_n_simulations(T, S, game,payoff_matrix):
    print("Running simulations for graph of game:{}, T: {}, S:{}".format(game,T,S))
    st = time.time()
  
    
    folder_name = "./data/lr_{}/game_{}/W_{}_N_{}_z_{}_betae_{}_betaa_{}/".format(lr,game,W,N,z,beta_e,beta_a)
    create_subfolders(folder_name)
    # for simulation in sim_generator:
    #     logging.info("Running Simulation no : {}".format(simulation))
    #     __play_game_for_g_generations(N, z, beta_e, beta_a, W, T, S, game, lr, simulation,folder_name, payoff_matrix)
    filename = os.path.join(folder_name,"_T_{}_S_{}_run.csv".format(T,S))
    if not os.path.exists(filename): 
           write_csv(filename, __HEADER_, mode="w")
           done_simulations = list()
    else:
        df = read_csv(filename)
        done_simulations = list(df["run_no"])

    remainder_simulations = list(set(range(__N_INDEPENDENT_SIMULATIONS_)) - set(done_simulations)) 
    print("!!! len remainder sim:{} T:{}, S:{}".format(len(remainder_simulations),T,S))
    if len(remainder_simulations) == 0: 
        return
   
    # print("sim generator:")
    # content = [Parallel(n_jobs=8, verbose=10)(delayed(__play_game_for_g_generations)(N, z, beta_e, beta_a, pr_W, T, S, game, lr, simulation,filename,payoff_matrix) for simulation in sim_generator)]
    # write_csv(filename, content[0])
    args = [(T, S, game, simulation,filename,payoff_matrix) for simulation in remainder_simulations]
    processes =  int(os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())) - 1
    print("No of processes: ", processes)
    # pool = ThreadPool(processes)
    # pool = Pool(processes)
    # print("[{}] No of parallel processes for simulations:{} ".format(os.environ["SLURMD_NODENAME"],processes))
    # results =  pool.starmap(__play_game_for_g_generations, args)
    write_csv(filename, results)
    pool.close()
    pool.join()
    print("Processed parallel sims job for T:{}, S:{} in time: {}".format(T,S,time.time()-st))


def __do_job(T, S, game):
    st = time.time()
    payoff_matrix = _get_payoff_matrix(T,S)        
    __play_game_for_n_simulations(T, S, game, payoff_matrix)
    print("Time taken for T:{}, S:{}, time: {}".format(T,S,time.time()-st))

def __play_game_per_Tchunk_and_Schunk(T_chunk, S_chunk, game):
    print("Processing chunk T: {}, S:{}".format(T_chunk,S_chunk))
    big_args_list = list()
    folder_name = "./data/lr_{}/game_{}/W_{}_N_{}_z_{}_betae_{}_betaa_{}/".format(lr,game,W,N,z,beta_e,beta_a)
    create_subfolders(folder_name)
    st = time.time()
    for T in T_chunk:
        for S in S_chunk:
            payoff_matrix = _get_payoff_matrix(T,S)
            filename = os.path.join(folder_name,"_T_{}_S_{}_run.csv".format(T,S))
            if not os.path.exists(filename): 
                write_csv(filename, __HEADER_, mode="w")
                done_simulations = list()
            else:
                df = read_csv(filename)
                done_simulations = list(df["run_no"])

            remainder_simulations = list(set(range(__N_INDEPENDENT_SIMULATIONS_)) - set(done_simulations)) 
            args = [(T, S, game, remainder_simulations,filename,payoff_matrix)]
            big_args_list.extend(args)
    processes =  multiprocessing.cpu_count() - 1
    
    print("[{}] No of parallel processes:{}  for args: {}".format(os.environ.get("SLURMD_NODENAME",""),processes,len(big_args_list)))
    # pool = Pool(processes)
    # pool.starmap(__play_game_for_g_generations, big_args_list)
    # pool.close()
    # pool.join()
    for arg in big_args_list:
        st2 = time.time()
        T, S, game, simulations,filename,payoff_matrix = arg
        __play_game_for_g_generations(T, S, game, simulations,filename,payoff_matrix) 



    print("Done Processing chunk T: {}, S:{} at time: {}".format(T_chunk,S_chunk, time.time()-st))


def __play__game(game, T_chunk, S_chunk):
    print("Playing game: ",game)
    st = time.time()
    # for T_chunk in T_chunks:
    #      for S_chunk in S_chunks:
    #         __play_game_per_Tchunk_and_Schunk(N, z, beta_e, beta_a, W, T_chunk, S_chunk, game, lr)
    # with parallel_backend('threading'):
    #      [Parallel(n_jobs=T_S_cores)(delayed(__play_game_per_Tchunk_and_Schunk)(T_chunk, S_chunk, game) for T_chunk in T_chunks for S_chunk in S_chunks)]
  
    # results = pool.starmap(__play_game_for_g_generations, args)

    
    __play_game_per_Tchunk_and_Schunk(T_chunk,S_chunk,game)
    # pool = Pool(4)
    # pool.starmap(__play_game_per_Tchunk_and_Schunk, args)
    # pool.close()
    # pool.join()
     
    print("Done playing game: {} for time: {}".format(game, time.time()-st))

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--W", help="Time scale Ratio", type=float, default=0.0)
    # parser.add_argument("--N", help="Number of Nodes", type=int, default=1000)
    # parser.add_argument("--z", help="Average Connectivity",  type=int, default=30)
    # parser.add_argument("--beta_e", help="Strategy Update", type=float, default=0.005)
    # parser.add_argument("--beta_a", help="Adjustment of Ties", type=float, default=0.005)
    parser.add_argument("--game", help="Type of Social Dilemma (sg, sh, pd)", type=str)
    parser.add_argument("--chunk", help="we divide 100 by 100 grid into n chunks", type=int)
    # parser.add_argument("--lr", help="Possible types of link recommenders - rewiring", type=str, default="rewiring")
    
    args = parser.parse_args()
    game = args.game
    
    T_chunk, S_chunk = chunk_dict[game][args.chunk]["T"], chunk_dict[game][args.chunk]["S"]
    T_chunk , S_chunk = [0], [0]
    print("Node - [{}], Simulations running on device: {} , and  game: {}, chunk: {}".format(os.environ.get("SLURMD_NODENAME",""),device, game, args.chunk))
    print("Number of Nodes: {}, z: {}".format(N,z))
    print("Beta e: {}, Beta a: {}, W: {}".format(beta_a,beta_e,W))
    print("Link Recommender: {}, game: {}".format(lr, game))
    __play__game(game, T_chunk, S_chunk)