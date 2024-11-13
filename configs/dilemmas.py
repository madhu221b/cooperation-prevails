"""
Creating a 2-D parameter space of T and S with fixed R and P

"""

import numpy as np

# Mutual Cooperation 
R = 1

# Mutual Defection 
P = 1

# The Gap of 0.1 is known from paper - Evolutionary Dynamics of Social Dilemmas in Structured Heterogeneous Populations
gap = 0.1

SNOWDRIFT_DICT = {
    "T": np.arange(1, 2+gap, gap),
    "S": np.arange(0, 1+gap, gap)
}

STAGHUNT_DICT = {
    "T": np.arange(0, 1+gap, gap),
    "S": np.arange(0, -1-gap, -gap)
}

PRISONERS_DILEMMA_DICT = {
    "T": np.arange(1, 2+gap, gap),
    "S": np.arange(0, -1-gap, -gap)
}

GAME_DICT = {
   "snowdrift": SNOWDRIFT_DICT,
   "staghunt": STAGHUNT_DICT,
   "pd": PRISONERS_DILEMMA_DICT
    
}

