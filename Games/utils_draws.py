import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

def single_arm_draw(mu,
                    horizon,
                    output_pickle=None,
                    law='Bernoulli',
                    plot=True):
    """
    draws_dict["draws_in_advance"] : (K, T)
    """
    mu_tile = np.tile(mu, (horizon, 1)).T
    draws_dict = {}
    draws_dict["mu"] = mu
    draws_dict["horizon"] = horizon
    draws_dict["draws_in_advance"] = np.random.binomial(1,mu_tile)
    if output_pickle:
        pickle.dump(draws_dict, open(output_pickle, 'wb'))
    if plot:
        plt.figure(figsize=(15,15))
        plt.imshow(draws_dict["draws_in_advance"], interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()
    return draws_dict


def pair_arm_draw(mu_row,
                  mu_col,
                  horizon,
                  output_pickle=None,
                  law='Bernoulli',
                  plot=True):
    nb_row, nb_col = len(mu_row), len(mu_col)
    total_pair_arms = nb_row * nb_col
    mu_matrix = np.dot(mu_row.reshape(nb_row,1), mu_col.reshape(1, nb_col))
    mu_flat = mu_matrix.flatten()
    draws_dict = single_arm_draw(mu=mu_flat,
                                 horizon=horizon,
                                 output_pickle=None,
                                 law=law,
                                 plot=plot)
    draws_dict['mu_row'] = mu_row
    draws_dict['mu_col'] = mu_col
    if law=='Bernoulli':
        draws_dict['draws_in_advance'] = np.array(draws_dict['draws_in_advance'] ,dtype='int16')
    if output_pickle:
        pickle.dump(draws_dict, open(output_pickle, 'wb'))
    return draws_dict




if __name__ == "__main__":
    horizon = 1000
    mu_row = np.linspace(0.1, 0.5, 5)
    mu_col = np.linspace(0.1, 0.3, 3)
    pair_arm_draw(mu_row=mu_row,
                  mu_col=mu_col,
                  horizon=horizon,
                  output_pickle="../test_folder/pair_draw_1000.p",
                  law='Bernoulli',
                  plot=False)

    mu = np.linspace(0.1, 0.5, 5)
    single_arm_draw(mu=mu,
                  horizon=horizon,
                  output_pickle="../test_folder/single_draw_1000.p",
                  law='Bernoulli',
                  plot=False)
