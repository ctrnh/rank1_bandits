import unittest
import numpy as np
import pickle as p
import utils_draws as ud
import sys
sys.path.append("..")
from Environments import Rank1Env as r1e


class TestGame(unittest.TestCase):
    def test_draws_single(self):
        mu = np.linspace(0.1,0.8,8)
        horizon = 10000
        draws_dict = ud.single_arm_draw(mu=mu,
                        horizon=horizon,
                        output_pickle=None,
                        law='Bernoulli',
                        plot=False)
        draws_in_advance = draws_dict["draws_in_advance"]
        self.assertEqual(draws_in_advance.shape, (len(mu), horizon))
        mean_draws = np.mean(draws_in_advance, axis=1)
        diff_mu = np.abs(mean_draws - mu)

        self.assertTrue((diff_mu<=0.02).all())
        self.assertTrue((diff_mu>0).all())


    def test_draws_pair(self):
        mu_row = np.linspace(0.1,0.8,8)
        mu_col = np.linspace(0.1,0.5,5)
        nb_row, nb_col = len(mu_row), len(mu_col)
        horizon = 10000
        draws_dict = ud.pair_arm_draw(mu_row=mu_row,
                          mu_col=mu_col,
                          horizon=horizon,
                          output_pickle=None,
                          law='Bernoulli',
                          plot=False)

        mu_matrix = np.dot(mu_row.reshape(nb_row,1), mu_col.reshape(1, nb_col))
        mu_flat = mu_matrix.flatten()

        draws_in_advance = draws_dict["draws_in_advance"]
        self.assertEqual(draws_in_advance.shape, (nb_row*nb_col, horizon))
        mean_draws = np.mean(draws_in_advance, axis=1)
        diff_mu = np.abs(mean_draws - mu_flat)
        self.assertTrue((diff_mu<=0.02).all())
        self.assertTrue((diff_mu>0).all())




if __name__ == '__main__':
    unittest.main()
