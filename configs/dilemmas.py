"""
Creating a 2-D parameter space of T and S with fixed R and P

"""

import numpy as np

# Mutual Cooperation 
R = 1

# Mutual Defection 
P = 0

# The Gap of 0.1 is known from paper - Evolutionary Dynamics of Social Dilemmas in Structured Heterogeneous Populations
gap = 0.1

SNOWDRIFT_DICT = {
    "T": [np.round(_,2) for _ in  np.arange(1, 2+gap, gap)],
    "S": [np.round(_,2) for _ in np.arange(0, 1+gap, gap)]
}

STAGHUNT_DICT = {
    "T": [np.round(_,2) for _ in np.arange(0, 1, gap)],
    "S": [np.round(_,2) for _ in np.arange(-0.1, -1-gap, -gap)]
}

PRISONERS_DILEMMA_DICT = {
    "T": [ np.round(_,2) for _ in np.arange(1, 2+gap, gap)],
    "S": [np.round(_,2) for _ in np.arange(-0.1, -1-gap, -gap)]
}

OTHER_DICT = {
    "T": [ np.round(_,2) for _ in np.arange(0, 1, gap)],
    "S": [np.round(_,2) for _ in np.arange(0, 1+gap, gap)]
}

GAME_DICT = {
   "sg": SNOWDRIFT_DICT,
   "sh": STAGHUNT_DICT,
   "pd": PRISONERS_DILEMMA_DICT,
   "other": OTHER_DICT
    }

CHUNK_DICT = {   # sg takes a lot of time so we make more chunks of sg
    "sg": 8,
   "sh": 4,
   "pd": 4,
   "other": 4
}

def get_stats(game_dict):
    stat_dict = dict()
    for game, param_range in game_dict.items():
        total_configs = len(param_range["T"])*len(param_range["S"])
        print("Game: {}, Total Configs: {}".format(game,total_configs))
        stat_dict[game] = total_configs


def get_chunks(game_dict, chunk_dict):
    chunk_dict = dict()
    for game, param_range in game_dict.items():
        T_range, S_range = param_range["T"], param_range["S"]
        T_chunks = np.array_split(T_range, CHUNK_DICT[game]//2)
        S_chunks = np.array_split(S_range, CHUNK_DICT[game]//2)
        id_to_chunkids = dict()
        i = 0
        chunk_dict[game] = dict()
        for  T_chunk in T_chunks:
            for  S_chunk in  S_chunks:
                chunk_dict[game][i] = {"T":T_chunk, "S":S_chunk}
                i += 1
    return chunk_dict

       

stat_dict = get_stats(GAME_DICT)

chunk_dict = get_chunks(GAME_DICT, CHUNK_DICT)




