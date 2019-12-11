import sys
sys.path.append("../")
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from Games import utils_draws as ud
from Games import UnimodalGame as ug

from Environments import Rank1Env as r1e
from Policies import OSUB

def draws_advance_pair(nb_row, nb_col, horizon, output_name, two_different_arms=None):
    if two_different_arms is None:
        mu_row = np.linspace(0.1, 0.9, nb_row)
        mu_col = np.linspace(0.1, 0.9, nb_col)
    else:
        mu_row = np.array([two_different_arms[0]] + [two_different_arms[1] for i in range(nb_row-1)])
        mu_col = np.array([two_different_arms[0]] + [two_different_arms[1] for i in range(nb_col-1)])
    draws_dict = ud.pair_arm_draw(mu_row=mu_row,
                      mu_col=mu_col,
                      horizon=horizon,
                      output_pickle=output_name,
                      law='Bernoulli',
                      plot=False)

if __name__ == "__main__":
    ## Parameters
    output_folder = "./draws"
    results_folder = "./results"

    horizons = [100000]
    nb_row = [32]
    nb_col = [32]
    two_different_arms = np.array([0.75, 0.25],dtype='float32') # first is optimal
    nb_draws = 1


    for h in horizons:
        for r in nb_row:
            for c in nb_col:
                if two_different_arms is not None:
                    prefix_name = f'two_diff_{two_different_arms[0]}-{two_different_arms[1]}'
                else:
                    prefix_name = 'pair_unif'
                name_new_folder = f'{prefix_name}_row-{r}_col-{c}_horizon-{h}'
                out_dir_mult_draws = os.path.join(output_folder, name_new_folder)
                if not os.path.exists(out_dir_mult_draws):
                    os.mkdir(out_dir_mult_draws)
                for i in tqdm(range(nb_draws)):
                    out_name = os.path.join(out_dir_mult_draws,f'{i}.p')
                    if not os.path.exists(out_name):
                        draws_advance_pair(r, c, h, out_name, two_different_arms)

                new_results_folder = os.path.join(results_folder, name_new_folder)
                if not os.path.exists(new_results_folder):
                    os.mkdir(new_results_folder)
