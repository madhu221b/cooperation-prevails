import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from configs.dilemmas import gap
from configs.game_configs import __N_INDEPENDENT_SIMULATIONS_

plot_directory = "./plots/"

def generate_heatmap(W, N, z):
    games = ["sh","sg","pd","other"]
    T_range = [np.round(_,2) for _ in np.arange(0, 2+gap, gap)]
    S_range = [np.round(_,2) for _ in np.arange(-1, 1+gap, gap)]
    T_dict = {val:i for i, val in enumerate(T_range)}
    S_dict = {val:i for i, val in enumerate(S_range)}
    x_ticks, y_ticks = T_range, S_range
    heatmap = np.zeros((len(T_range), len(S_range)))
    cmap = "Spectral_r"
    
    x_ticks = [str(x) if i in [0,10,20] else '' for i,x in enumerate(T_range)]
    y_ticks = [str(x) if i in [0,10,20] else '' for i,x in enumerate(S_range)]
    

    for game in games:
        configs = 0
        folder_name = "./data/lr_rewiring/game_{}/W_{}_N_{}_z_{}_betae_0.005_betaa_0.005".format(game,float(W),N,z)
        if not os.path.exists(folder_name): continue
        for file_name in os.listdir(folder_name):
            csv_file = os.path.join(folder_name,file_name)
            try:
               df = pd.read_csv(csv_file)
            except Exception as e:
                continue
            if df[df.duplicated(["run_no"])].shape[0] > 0:
                print("!!! ", file_name)
            if  df.shape[0] == __N_INDEPENDENT_SIMULATIONS_: # ran for N independent simulations
                _,_, T,_, S, _ = file_name.split("_")
                T, S = float(T), float(S)
                row_idx, col_idx = S_dict[S], T_dict[T]
                avg_val = df["coop_frac"].mean()
                heatmap[row_idx][col_idx] = avg_val
                configs += 1
        print("Game : {} , configs : {}".format(game,configs))
                
                

    ax = sns.heatmap(heatmap, cmap=cmap, xticklabels=x_ticks, yticklabels=y_ticks)
    ax.invert_yaxis()
    ax.set(xlabel="T", ylabel="S")
    fig = ax.get_figure()
    fig.savefig(plot_directory+"trial_W_{}_N_{}_z_{}.pdf".format(W,N,z),bbox_inches='tight')
    ax.clear()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", help="Timescale", type=float, default=0)
    parser.add_argument("--N", help="Number of nodes", type=int, default=1000)
    parser.add_argument("--z", help="Avg. degree", type=int, default=30)
  
    args = parser.parse_args()
    generate_heatmap(args.W, args.N, args.z) 
