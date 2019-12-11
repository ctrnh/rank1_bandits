import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import os
from Games import utils_draws as ud
from Games import UnimodalGame as ug

from Environments import Rank1Env as r1e
from Policies import OSUB
import pickle as p
from tools.tools import klBern

def LB_two_diff_arms(two_different_arms,horizon, nb_row,nb_col,t=None):
    mu_opt = two_different_arms[0]
    mu_i = two_different_arms[1]
    if t:
        return (mu_opt-mu_i)*(nb_row + nb_col - 2)* np.log(t) / klBern(0.2,0.5)
    else:
        return (mu_opt-mu_i)*(nb_row + nb_col - 2)* np.log(horizon) / klBern(0.2,0.5)

if __name__ == "__main__":
    ## Parameters
    log_scale=1
    results_dir = "./results"
    algo_name = ['UTS',
                'OSUB',
                'KLUCB',
                #'Rank1ElimKL'
                ]
    horizon = 300000
    nb_row =  4
    nb_col =  4
    two_different_arms =  [0.75,0.25]

    if two_different_arms:
        prefix_name = f'two_diff_{two_different_arms[0]}-{two_different_arms[1]}'
    else:
        prefix_name = 'pair_unif'

    env_folder_name = f'{prefix_name}_row-{nb_row}_col-{nb_col}_horizon-{horizon}'
    root_folder = os.path.join(results_dir, env_folder_name)
    list_folders_to_plot = [os.path.join(root_folder, "UTS", "dle_2"),
                            #os.path.join(root_folder, "UTS", "dle_no_leader"),
                            os.path.join(root_folder, "OSUB"),
                            os.path.join(root_folder, "KLUCB"),
                            #os.path.join(root_folder, "Rank1ElimKL")
                            ]






    plt.figure(figsize=(18,15))
    for cur_folder in list_folders_to_plot:
        cur_path = cur_folder
        list_regrets = []
        for i_pkl, cur_pkl in enumerate(os.listdir(cur_path)):
            print(cur_path, cur_pkl)
            with open(os.path.join(cur_path, cur_pkl),'rb') as f:
                cur_pkl_regret = p.load(f)
            list_regrets.append(cur_pkl_regret)

        list_regrets = np.array(list_regrets)
        mean_regrets = np.mean(list_regrets, axis=0)
        std_regrets = np.std(list_regrets, axis=0)
        # upper_regrets = mean_regrets + 2*std_regrets
        # lower_regrets = mean_regrets - 2*std_regrets
        upper_regrets = np.percentile(list_regrets, 90, axis=0)
        lower_regrets = np.percentile(list_regrets, 10, axis=0)
        plt.plot(mean_regrets, '--',label=cur_folder)


        #plt.plot(upper_regrets, label="upper")
        #plt.plot(lower_regrets, label="lower")
        plt.fill_between((np.arange(horizon)),lower_regrets, upper_regrets, alpha=0.8)
    plt.plot([LB_two_diff_arms(two_different_arms,horizon,nb_row,nb_col,t) for t in range(horizon)], label="lb")
    if log_scale:
        plt.xscale("log")
        fig_name = f"{env_folder_name}_{'-'.join(algo_name)}_log.png"
    else:
        fig_name = f"{env_folder_name}_{'-'.join(algo_name)}.png"
    plt.legend()
    plt.title(fig_name)
    plt.savefig(os.path.join(root_folder,fig_name), format="png")
    plt.figure()
    plt.plot(std_regrets)
    plt.show()
