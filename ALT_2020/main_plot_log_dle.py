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
import seaborn as sns

def LB_two_diff_arms(two_different_arms,horizon, nb_row,nb_col,t=None):
    mu_opt = two_different_arms[0]
    mu_i = two_different_arms[1]
    if t:
        return (mu_opt-mu_i)*(nb_row + nb_col - 2)* np.log(t) / klBern(0.25,0.75)
    else:
        return (mu_opt-mu_i)*(nb_row + nb_col - 2)* np.log(horizon) / klBern(0.25,0.75)

if __name__ == "__main__":
    ## Parameters
    results_dir = "./results"
    horizon = 300000
    nb_row =  4
    nb_col =  nb_row
    two_different_arms =  [0.75,0.25]
    #dle_to_plot=[2,5,10,20,"infinity"]
    colors=sns.color_palette("dark")
    styles=['x','o','s','*','D','v','^','<','>','1','2','3','4','8']
    plt.rcParams.update({'font.size': 36})
    if two_different_arms:
        prefix_name = f'two_diff_{two_different_arms[0]}-{two_different_arms[1]}'
    else:
        prefix_name = 'pair_unif'

    env_folder_name = f'{prefix_name}_row-{nb_row}_col-{nb_col}_horizon-{horizon}'
    root_folder_to_plot = os.path.join(results_dir, env_folder_name,"UTS")
    list_folders_to_plot = [fn for fn in os.listdir(root_folder_to_plot)
                                if os.path.isdir(os.path.join(root_folder_to_plot, fn))]
                                #and int(fn.split('_')[1]) in dle_to_plot]

    plt.figure(figsize=(18,15))
    for i_folder,cur_folder in enumerate(list_folders_to_plot):
        cur_path = os.path.join(root_folder_to_plot, cur_folder)
        list_regrets = []
        for i_pkl, cur_pkl in enumerate(os.listdir(cur_path)):
            print(cur_pkl, cur_folder)
            with open(os.path.join(cur_path, cur_pkl),'rb') as f:
                cur_pkl_regret = p.load(f)
            list_regrets.append(cur_pkl_regret)

        list_regrets = np.array(list_regrets)
        mean_regrets = np.mean(list_regrets, axis=0)
        std_regrets = np.std(list_regrets, axis=0)
        # upper_regrets = mean_regrets + 2*std_regrets
        # lower_regrets = mean_regrets - 2*std_regrets
        upper_regrets = np.quantile(list_regrets, 0.95, axis=0)
        lower_regrets = np.quantile(list_regrets, 0.05, axis=0)
        upper_regrets = np.percentile(list_regrets, 90, axis=0)
        lower_regrets = np.percentile(list_regrets, 10, axis=0)
        plt.plot(mean_regrets, label="gamma = "+cur_folder.split("_")[1],linewidth=2.5,
                color=colors[i_folder],#[len(list_folders_to_plot)-i_folder],
                marker=styles[i_folder],
                 markevery=0.3, ms=6)


        #plt.plot(upper_regrets, label="upper")
        #plt.plot(lower_regrets, label="lower")
        plt.fill_between((np.arange(horizon)),lower_regrets, upper_regrets, alpha=0.15)
    #plt.plot([LB_two_diff_arms(two_different_arms,horizon,nb_row,nb_col) for t in range(horizon)], label="lb")
    #plt.xscale("log")
    #sns.color_palette('deep')

    plt.legend(loc=2)
    plt.ylabel('cumulative regret')
    plt.xlabel('time steps')
    #plt.title(f'K = L = {nb_col}, horizon = {horizon}, mu* = 0.75, mu_i = 0.25')
    plt.savefig(os.path.join(root_folder_to_plot,env_folder_name+"log.png"), format="png")
    #plt.figure()
    #plt.plot(std_regrets)
    plt.show()
