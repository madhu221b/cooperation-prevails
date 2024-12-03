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
W = 1.0
pr_W = _get_pr_of_strategy_update(W)
pr_Ws = torch.tensor([pr_W,1-pr_W],device=device) # __EVENTS_ = = ["strategy","structural"]
lr = "rewiring"



def _is_convergence_reached(is_convergence_all, node_attrs_all, g, filename):
    is_all_sims = False
      
    non_convergence_idxs = is_convergence_all[is_convergence_all[:,1] == 0][:,0]
    non_convergence_idxs = non_convergence_idxs.to(device)
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

    file_name_snap = filename.replace("lr_rewiring", "snaps_rewiring").replace(".csv",".pkl")

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
         
        print("generation  g: {} , Non Converged Sims: {} ".format(g,remainder_sims))
   
        ##Shuffling of Nodes
        shuffled_nodes = torch.stack([torch.randperm(N) for _ in range(remainder_sims)])
       
        
        pr_Ws_tensor = pr_Ws.repeat(remainder_sims,1) # n_sims X 2
    
        events = torch.multinomial(pr_Ws_tensor,N,replacement=True)  # n_sims X N
        
       
        for i in range(N): # select one agent iteratively out of N agents
            a_s = shuffled_nodes[:,i] # select source nodes for all simulations # sim
            event_s = events[:,i]
            
            # 1.1 ngh_a = find neighbors of a, 
            ngh_dict = dict() # sim no -> k , ngh_a, ngh_b -> val
            sim_ab_tensor = torch.zeros((remainder_sims,4),dtype=int, device=device) # sim_no, a, b, node_attr_b
            
        
            bs = (adj_matrix_all[non_convergence_idxs,a_s,:] == 1).nonzero()
         
            # print(a_s)
            # print(non_convergence_idxs)
            for i, (sim_no, a) in enumerate(zip(non_convergence_idxs, a_s)):         
                potential_bs = bs[bs[:,0] == i,1]
                rand_int = torch.randint(0, potential_bs.size(0), (1,))[0]
                b = potential_bs[rand_int]
                ngh_bs =  (adj_matrix_all[sim_no,b,:] == 1).nonzero()
                ngh_dict[int(sim_no)] = {"ngh_a":potential_bs, "ngh_b":ngh_bs, "a": a, "b": b} 
                sim_ab_tensor[i] = torch.tensor([sim_no, a, b, node_attrs_all[sim_no][b]])

                
            
            # 2. do strategy events for nodes of all simulations
            event_struct = non_convergence_idxs[event_s == 0]
            if event_struct.size(0) != 0: 
                payoff_tensor = _get_payoff_of_nodes_for_all_sims(ngh_dict, node_attrs_all, payoff_matrix, event_struct) # sim_nos x 3
                pr_strategy_replace = _get_pr_of_strategy_replacement(payoff_tensor[:,1], payoff_tensor[:,2], beta_e) # sim_nos X 1
                prs_str = torch.cat((pr_strategy_replace[:, None], (1-pr_strategy_replace)[:, None]), dim=-1)
                replace_idx = torch.multinomial(prs_str, 1).squeeze(1) # sim_nos X 1
                sims_replace = event_struct[replace_idx == 0] # replace idxs = 0 , pr_strategy_replace selected
                mask = torch.isin(sim_ab_tensor[:, 0], sims_replace)
                sim_ab_tensor_subset = sim_ab_tensor[mask]
                a_replace = sim_ab_tensor_subset[:, 1]
                node_attrs_b_replace = sim_ab_tensor_subset[:, 3]
                node_attrs_all[sims_replace, a_replace] =  node_attrs_b_replace
            
            # 3. do structural events for nodes of all simulations
            event_strategy = non_convergence_idxs[event_s == 1]
            if event_strategy.size(0) != 0:
                # Get all simulations for event strategy
                mask = torch.isin(sim_ab_tensor[:, 0], event_strategy)
                sim_ab_tensor_subset = sim_ab_tensor[mask]
                # Get all simulations where node b is defector
                sim_ab_tensor_subset = sim_ab_tensor_subset[sim_ab_tensor_subset[:,3] == 0, :]
                strategy_sims = sim_ab_tensor_subset[:,0]

                payoff_tensor = _get_payoff_of_nodes_for_all_sims(ngh_dict, node_attrs_all, payoff_matrix, strategy_sims)
                pr_adjust_ties = _get_pr_of_adjust_ties(payoff_tensor[:,1], payoff_tensor[:,2], beta_a)
                prs_adjust = torch.cat((pr_adjust_ties[:, None], (1-pr_adjust_ties)[:, None]), dim=-1)
                adjust_idx = torch.multinomial(prs_adjust, 1).squeeze(1) # sim_nos X 1
                sims_adjust = strategy_sims[adjust_idx == 0] # adjust idxs = 0 , pr_strategy_adjust selected
                
                sim_update, a_update, b_update, c_update = [], [], [], []
                for sim_no in sims_adjust:
                    sim_no = int(sim_no)
                    ngh_a, ngh_b = ngh_dict[sim_no]["ngh_a"], ngh_dict[sim_no]["ngh_b"]
                    a, b = ngh_dict[sim_no]["a"], ngh_dict[sim_no]["b"]
                    candidates = np.setdiff1d(np.setdiff1d(ngh_b, ngh_a), a) # neighbours of b and not neighbors of a
                    if len(candidates) != 0:   # "b and a have exclusively different neighbors"
                        rand_int = torch.randint(0, len(candidates), (1,))[0]
                        c = candidates[int(rand_int)] # select a candidate to rewire
                        # adj_matrix_all[sim_no, a, b] = adj_matrix_all[sim_no, b, a] = 0
                        # adj_matrix_all[sim_no, a, c] = adj_matrix_all[sim_no, c, a] = 1
                        sim_update.append(sim_no)
                        a_update.append(a)
                        b_update.append(b)
                        c_update.append(c)

                adj_matrix_all[sim_update,a_update,b_update] = adj_matrix_all[sim_update,b_update,a_update] = 0    
                adj_matrix_all[sim_update,a_update,c_update] = adj_matrix_all[sim_update,c_update,a_update] = 1        
            

        print("time for 1 generation: ", time.time()-st)
        if g % 1000 == 0:
            save_generation_snap(node_attrs_all, adj_matrix_all, g, file_name_snap)
        if g >= (__N_GENERATIONS_ - OFFSET):
            print("Cumulating the frac at generation g: ", g)
            is_convergence_all = get_fracs_at_g(node_attrs_all, non_convergence_idxs, is_convergence_all)



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
    pool = Pool(processes)
    pool.starmap(__play_game_for_g_generations, big_args_list)
    pool.close()
    pool.join()

    # for arg in big_args_list:
    #     st2 = time.time()
    #     T, S, game, simulations,filename,payoff_matrix = arg
    #     __play_game_for_g_generations(T, S, game, simulations,filename,payoff_matrix) 



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
    print("Node - [{}], Simulations running on device: {} , and  game: {}, chunk: {}".format(os.environ.get("SLURMD_NODENAME",""),device, game, args.chunk))
    print("Number of Nodes: {}, z: {}".format(N,z))
    print("Beta e: {}, Beta a: {}, W: {}".format(beta_a,beta_e,W))
    print("Link Recommender: {}, game: {}".format(lr, game))
    __play__game(game, T_chunk, S_chunk)