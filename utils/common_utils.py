import random
import os
import pickle
import numpy as np
import pandas as pd
import csv
import torch

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_subfolders(fn):
    
    path = os.path.dirname(fn)
    os.makedirs(path, exist_ok = True)

def save_pickle(obj, fn):
    try:
        create_subfolders(fn)
        with open(fn,'wb') as f:
            pickle.dump(obj, f)
        print('{} saved!'.format(fn))
    except Exception as ex:
        print(ex)

def read_pickle(fn):
    obj = None
    try:
        with open(fn,'rb') as f:
            obj = pickle.load(f)
    except Exception as ex:
        print(ex)
    return obj

def write_csv(filename, content, mode="a"):
    logging.info("[{}] Writing to csv: {} ".format(os.environ.get("SLURMD_NODENAME","") ,filename))
    with open(filename, mode) as f:
            writer = csv.writer(f)
            if mode == "a":
               writer.writerow(content)
            else:
                writer.writerow(content)

def read_csv(filename):
    df = pd.read_csv(filename)
    return df
