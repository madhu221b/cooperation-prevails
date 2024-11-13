import argparse
from tqdm import tqdm
import numpy as np

from utils. common_utils import create_subfolders
from utils.game_utils import _is_reach_convergence,  _get_most_recent_generation, _get_pr_of_strategy_update

from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_, __N_GENERATIONS_, __EVENTS_
from configs.dilemmas import GAME_DICT

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

def __play_game_for_one_generation(graph, beta, pr_strategy, lr, T, S):
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)

    for node in nodes:
        event = np.random.choice(__EVENTS_, size=1, p=[pr_strategy,1-pr_strategy])[0]
        if event == "strategy":

    
    return

def __play_game_for_g_generations(N, z, beta, W, T, S, game, lr, simulation):
    graph, gen_no = _get_most_recent_generation(N, z, beta, W, T, S, game, lr, simulation)
    pr_strategy = _get_pr_of_strategy_update(W)

    if _is_reach_convergence(graph):
        print("Convergence was reached at gen_no: {}".format(gen_no))
        return
    
    gen_generator =  tqdm(range(gen_no+1, __N_GENERATIONS_), desc='Running Generations')
    for g in gen_generator:
        graph = __play_game_for_one_generation(graph, beta, pr_strategy, lr, T, S)
        if _is_reach_convergence(graph):
            logging.info("Convergence is reached")
            print("##TODO more steps")
            break


def __play_game_for_n_simulations(N, z, beta, W, T, S, game, lr):
    print("Running {} simulations for graph of N: {}, z:{}, game:{}, T: {}, S:{}".format(__N_INDEPENDENT_SIMULATIONS_,N,z,game,T,S))
    for simulation in range(__N_INDEPENDENT_SIMULATIONS_):
        logging.info("Running Simulation no : {}".format(simulation))
        __play_game_for_g_generations(N, z, beta, W, T, S, game, lr, simulation)

def __play__game(N, z, beta, game, W, lr):
    T_range, S_range = GAME_DICT[game]["T"], GAME_DICT[game]["S"]

    for T in T_range:
        for S in S_range:
            __play_game_for_n_simulations(N, z, beta, W, T, S, game, lr)
            
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", help="Time scale Ratio", type=float, default=0.0)
    parser.add_argument("--N", help="Number of Nodes", type=int, default=1000)
    parser.add_argument("--z", help="Average Connectivity",  type=int, default=30)
    parser.add_argument("--beta", help="Inverse Temperature of Selection", type=float, default=0.005)
    parser.add_argument("--game", help="Type of Social Dilemma (sg, sh, pd)", type=str)
    parser.add_argument("--lr", help="Possible types of link recommenders - rewiring", type=str, default="rewiring")


    args = parser.parse_args()

    __play__game(args.N, args.z, args.beta, args.game, args.W, args.lr)