import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from Games.Game import Game

import os
import pic        policy = self.policykle as p
from tqdm import tqdm

class Rank1ElimGame(Game):
    def __init__(self,
                environment,
                policy,
                horizon):

        super().__init__(environment,
                        policy,
                        horizon)
		self.delta_tilde = 1
		self.t_l = [0, 0]
		self.l = -1
        self.nb_row = self.env.nb_row
        self.nb_col = self.env.nb_col


        self.h_u = np.arange(self.env.nb_row)
        self.h_v = np.arange(self.env.nb_col)

        self.row_reward =  np.zeros(self.mu_matrix.shape)
        self.col_reward =  np.zeros(self.mu_matrix.shape)

        self.UB_rows = np.zeros((self.nb_row))
        self.LB_rows = np.zeros((self.nb_row))

        self.UB_cols = np.zeros((self.nb_col))
        self.LB_cols = np.zeros((self.nb_col))

        self.eps = 1


    def playGame(self, save_regret_every=1, save_arm_drawn=False):
        t = 0
        while t < self.horizon:
            self.l += 1
            self.t_l[1] = np.ceil(16 * np.log(self.horizon) / (self.delta_tilde ** 2))

            active_rows = (np.arange(self.nb_row) == self.h_u).astype(int)
            active_cols = (np.arange(self.nb_col) == self.h_v).astype(int)

            for k in range(int(self.t_l[1] - self.t_l[0])):
                self.t_l[0] = self.t_l[1]
                chosen_col_idx = np.random.randint(0, self.nb_col)
                j = self.h_v[chosen_col_idx]
                for i in np.argwhere(active_rows == 1).flatten():
                    if t <= self.T:
                        cur_arm = self.env.get_arm_idx((i,j))
                        reward_t = cur_arm.draw(t)
                        self.row_reward[i,j] += reward_t
                        regret_t = self.opt_arm.mu - cur_arm.mu
                        if t%save_regret_every==0:
                            self.regret_history.append(regret_t)
                    else:
                        break


            t += 1
        regret_t = 0
        for t in tqdm(range(self.horizon)):
            arm_t, reward_t = self.policy.playArm(self.env,
                                                            t)
            regret_t += self.opt_arm.mu - arm_t.mu

            if save_arm_drawn:
                self.arm_drawn_history.append(arm_t)
            if t%save_regret_every==0:
                self.regret_history.append(regret_t)
