import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
import torch
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool
from utils.common_utils import create_subfolders, write_csv, read_csv, read_pickle, save_generation_snap
from utils.game_utils import _is_reach_convergence,  _get_init_graphs, \
              _get_pr_of_strategy_update, _get_payoff_of_node, _get_payoff_of_nodes, _get_payoff_of_nodes_for_all_sims, \
               _get_pr_of_strategy_replacement, \
            _get_pr_of_adjust_ties, do_rewiring, _get_payoff_matrix, get_fracs_at_g

from utils.rewiring_utils import get_common_rewiring_candidate

from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_, __N_GENERATIONS_, __EVENTS_, __HEADER_, OFFSET
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
W =  8.0
pr_W = _get_pr_of_strategy_update(W)
pr_Ws = torch.tensor([pr_W,1-pr_W],device=device) # __EVENTS_ = = ["strategy","structural"]
lr = "nopr_rewiring" # common_ngh, rewiring



def _is_convergence_reached(is_convergence_all, node_attrs_all, g, filename):
    is_all_sims = False
      
    non_convergence_idxs = is_convergence_all[is_convergence_all[:,1] == 0][:,0]
    remainder_sims = non_convergence_idxs.size(0)
    if remainder_sims == 0:
        print("Convergence reached for all sims")
        is_all_sims = True
     
    sum_c = torch.sum(node_attrs_all[non_convergence_idxs], dim=-1)
    frac_c = (sum_c/N)
    sim_frac = torch.cat((non_convergence_idxs[:, None], frac_c[:, None]), dim=-1)
    mask = torch.logical_or(sim_frac[:, 1] == 0, sim_frac[:, 1] == 1)
    sim_frac = sim_frac[mask]
    
    if sim_frac.size(0) > 0: # check for cooperators/defectors
        converge_sims = sim_frac[:,0]
        mask = torch.isin(is_convergence_all[:, 0], converge_sims)
        converge_idxs = torch.nonzero(mask)
        is_convergence_all[converge_idxs, 1] = 1

        non_convergence_idxs = is_convergence_all[is_convergence_all[:,1] == 0][:,0]
        remainder_sims = non_convergence_idxs.size(0)
        print("remainder sims after convergence updation: ", remainder_sims)
        results = [[int(sim), float(frac), g] for sim, frac in sim_frac]
        write_csv(filename, results)

        if remainder_sims == 0: is_all_sims = True
    
    if   g == __N_GENERATIONS_ - 1: # we have reached last generation
        print("Cumulating results for the last generation: ", g)
        mask = torch.isin(is_convergence_all[:, 0], non_convergence_idxs)
        idxs = torch.nonzero(mask)
        frac_c = is_convergence_all[idxs, 1]/OFFSET
        sim_frac = torch.cat((is_convergence_all[idxs, 0][:, None], frac_c[:, None]), dim=-1)


        is_convergence_all[idxs, 1] = 1
        results = [[int(sim), float(frac), g] for sim, frac in sim_frac]
        write_csv(filename, results)

        non_convergence_idxs = is_convergence_all[is_convergence_all[:,1] == 0][:,0]
        remainder_sims = non_convergence_idxs.size(0)
        if remainder_sims == 0: is_all_sims = True

    return is_all_sims, remainder_sims, non_convergence_idxs, is_convergence_all

def __play_game_for_g_generations(T, S, game, simulations,filename,payoff_matrix): 
   
    file_name_snap = filename.replace("lr_{}".format(lr), "snaps_{}".format(lr)).replace(".csv",".pkl")
    file_name_snap = file_name_snap.replace("./data","/var/scratch/mmpawar")
    n_simulations = len(simulations)
    is_convergence_all = torch.zeros((n_simulations,2), device=device, dtype=int) # we maintain for all simulations, whether we reached convergence here
    is_convergence_all[:,0] = torch.tensor(simulations, dtype=int)
    
    print("T: {}, S:{} - Running  len Simulations  : {}".format(T,S, len(simulations)))
    
     # shape of node_attrs = [n_simulations, N] adj_matrix = [n_simulations,N,N]
    if not os.path.exists(file_name_snap):
         node_attrs_all, adj_matrix_all = _get_init_graphs(N, z, simulations)
         start_gen = 0
    else: # load the last most saved generation
          print("Loading the generations from :", file_name_snap)
          td = read_pickle(file_name_snap)
          node_attrs_all, adj_matrix_all = td["node_attrs_all"], td["adj_matrix_all"]
          start_gen = td["g"]+1
          print("Starting from generation: ", start_gen)
    
    

    for g in range(start_gen, __N_GENERATIONS_):  
        st = time.time()
        
        is_all_sims, remainder_sims, non_convergence_idxs, is_convergence_all = _is_convergence_reached(is_convergence_all, node_attrs_all, g, filename)
        if is_all_sims: break
         
        print("T: {}, S: {}, generation  g: {} , Non Converged Sims: {} ".format(T, S, g,remainder_sims))
   
        ##Shuffling of Nodes
        shuffled_nodes = torch.stack([torch.randperm(N) for _ in range(remainder_sims)])
       
        
        pr_Ws_tensor = pr_Ws.repeat(remainder_sims,1) # n_sims X 2
    
        events = torch.multinomial(pr_Ws_tensor,N,replacement=True)  # n_sims X N
        
       
        for i in range(N): # select one agent iteratively out of N agents
            a_s = shuffled_nodes[:,i] # select source nodes for all simulations # sim
            event_s = events[:,i]
            
            # 1.1 ngh_a = find neighbors of a, 
            sim_ab_tensor = torch.zeros((remainder_sims,3),dtype=int, device=device) # sim_no, a, b, node_attr_b
            sim_ab_tensor[:, 0] = non_convergence_idxs
            sim_ab_tensor[: , 1] = a_s
        
            bs = (adj_matrix_all[non_convergence_idxs,a_s,:] == 1).nonzero()
        
            for i, _ in enumerate(sim_ab_tensor):         
                potential_bs = bs[bs[:,0] == i,1]
                rand_int = torch.randint(0, potential_bs.size(0), (1,))[0]
                b = potential_bs[rand_int]
                sim_ab_tensor[i, 2] = b
            

                
            # 2. do strategy events for nodes of all simulations
            event_strategy = non_convergence_idxs[event_s == 0]
            if event_strategy.size(0) != 0: 
                payoff_tensor = _get_payoff_of_nodes_for_all_sims(nodes, node_attrs_all, adj_matrix_all, payoff_matrix, event_strategy, sim_ab_tensor) # sim_nos x 3
                pr_strategy_replace = _get_pr_of_strategy_replacement(payoff_tensor[:,1], payoff_tensor[:,2], beta_e) # sim_nos X 1
                
                prs_str = torch.cat((pr_strategy_replace[:, None], (1-pr_strategy_replace)[:,None]), dim=-1)
                replace_idx = torch.multinomial(prs_str, 1).squeeze(1) # sim_nos X 1
                sims_replace = event_strategy[replace_idx == 0] # replace idxs = 0 , pr_strategy_replace selected
                sim_ab_tensor_subset = sim_ab_tensor[torch.isin(sim_ab_tensor[:, 0], sims_replace)]
                a_replace = sim_ab_tensor_subset[:, 1]
                b_replace = sim_ab_tensor_subset[:, 2]
                node_attrs_all[sims_replace, a_replace] =  node_attrs_all[sims_replace, b_replace]
            
            # 3. do structural events for nodes of all simulations
            event_struct = non_convergence_idxs[event_s == 1]
            if event_struct.size(0) != 0:
                # Get all simulations for event strategy
       
                sim_ab_tensor_subset = sim_ab_tensor[torch.isin(sim_ab_tensor[:, 0], event_struct)]
                # Get all simulations where node b is defector
                sim_ab_tensor_subset = sim_ab_tensor_subset[node_attrs_all[sim_ab_tensor_subset[:,0], sim_ab_tensor_subset[:,2]] == 0, :]
                struct_sims = sim_ab_tensor_subset[:,0]

                payoff_tensor = _get_payoff_of_nodes_for_all_sims(nodes, node_attrs_all, adj_matrix_all, payoff_matrix, struct_sims, sim_ab_tensor_subset)
                if lr == "rewiring":
                    pr_adjust_ties = _get_pr_of_adjust_ties(payoff_tensor[:,1], payoff_tensor[:,2], beta_a)
                    prs_adjust = torch.cat((pr_adjust_ties[:, None], (1-pr_adjust_ties)[:, None]), dim=-1)
                    adjust_idx = torch.multinomial(prs_adjust, 1).squeeze(1) # sim_nos X 1
                    sims_adjust = struct_sims[adjust_idx == 0] # adjust idxs = 0 , pr_strategy_adjust selected
                    sim_ab_tensor_subset = sim_ab_tensor_subset[torch.isin(sim_ab_tensor_subset[:, 0], sims_adjust)]
                    
                sims_adjust, a_adjust, b_adjust =   sim_ab_tensor_subset[:, 0],  sim_ab_tensor_subset[: ,1],  sim_ab_tensor_subset[:, 2]

               
                
                diff_matrix = adj_matrix_all[sims_adjust, b_adjust] - adj_matrix_all[sims_adjust, a_adjust] # 1 - 0, 1 -1 
                c_s = (diff_matrix == 1).nonzero()
           
                sim_update, a_update, b_update, c_update = [], [], [], []
                for i, sim_no in enumerate(sims_adjust):
                    a = a_adjust[i]
                    potential_cs = c_s[c_s[:,0] == i,1]
                    potential_cs = potential_cs[potential_cs != a]
                    potential_cs_size = potential_cs.size(0)
                   
                   
                    if potential_cs_size != 0:   # "b and a have exclusively different neighbors"
                        if lr == "rewiring" or lr == "nopr_rewiring":
                           rand_int = torch.randint(0, potential_cs_size, (1,))[0]
                           c = potential_cs[rand_int] # select a candidate to rewire
                        elif lr == "common_ngh":
                            c = get_common_rewiring_candidate(a, potential_cs, adj_matrix_all[sim_no])

                        sim_update.append(sim_no)
                        a_update.append(a)
                        b_update.append(b_adjust[i])
                        c_update.append(c)

                adj_matrix_all[sim_update,a_update,b_update] = adj_matrix_all[sim_update,b_update,a_update] = 0    
                adj_matrix_all[sim_update,a_update,c_update] = adj_matrix_all[sim_update,c_update,a_update] = 1        
            

        print("T: {}, S: {} . time for 1 generation: {}".format(T,S, time.time()-st))
        if g % 1000 == 0 and g != 0:
            save_generation_snap(node_attrs_all, adj_matrix_all, g, file_name_snap)
        if g >= (__N_GENERATIONS_ - OFFSET):
            print("Cumulating the frac at generation g: ", g)
            is_convergence_all = get_fracs_at_g(node_attrs_all, non_convergence_idxs, is_convergence_all)
  


def __play_game_per_Tchunk_and_Schunk(T_chunk, S_chunk, game):
    # try:
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

    # processes =  multiprocessing.cpu_count() 
    # processes = 8
    # print("[{}] No of parallel processes:{}  for args: {}".format(os.environ.get("SLURMD_NODENAME",""),processes,len(big_args_list)))
    # pool = Pool(processes)
    # pool.starmap(__play_game_for_g_generations, big_args_list)
    # pool.close()
    # pool.join()

    for arg in big_args_list:
        st2 = time.time()
        T, S, game, simulations,filename,payoff_matrix = arg
        __play_game_for_g_generations(T, S, game, simulations,filename,payoff_matrix) 



    print("Done Processing chunk T: {}, S:{} at time: {}".format(T_chunk,S_chunk, time.time()-st))
    # except Exception as e:
    #     print("Error: ", filename, e)

def __play__game(game, T_chunk, S_chunk):
    print("Playing game: ",game)
    st = time.time()
    __play_game_per_Tchunk_and_Schunk(T_chunk,S_chunk,game)     
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
    # T_chunk, S_chunk = [1.0], [-0.1]
    print("Node - [{}], Simulations running on device: {} , and  game: {}, chunk: {}".format(os.environ.get("SLURMD_NODENAME",""),device, game, args.chunk))
    print("Number of Nodes: {}, z: {}".format(N,z))
    print("Beta e: {}, Beta a: {}, W: {}".format(beta_a,beta_e,W))
    print("Link Recommender: {}, game: {}".format(lr, game))
    __play__game(game, T_chunk, S_chunk)