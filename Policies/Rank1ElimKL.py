import numpy as np
import matplotlib.pyplot as plt
import tools
import Algorithm as a


class Rank1ElimKL(a.Algorithm):
    def __init__(self, environment, draws, is_updated=False):
        a.Algorithm.__init__(self, environment, draws, is_updated=False)
        self.L = self.environment.L
        self.M = self.environment.M

        self.matrix_uv = self.environment.matrix_uv
        self.matrix_uv_flat = self.environment.matrix_uv_flat

    def run_algo(self):

        t = 1
        delta_tilde = 1
        l = -1  # starts at l = 0
        t_l = [0, 0]  # init t_l = [t_{l-1},t_l], here t_0 = [t_-1,t_0] = [0,0]

        row_reward = np.zeros(self.matrix_uv.shape)
        col_reward = np.zeros(self.matrix_uv.shape)
        cumrewards_per_arm = np.zeros(self.K)  # cumulative rewards per arm (flatten idx)
        mu_hat = np.zeros(self.K)  # mu_hat per arm (flatten idx)

        h_u = np.arange(self.L)
        h_v = np.arange(self.M)

        UB_rows = np.zeros((self.L,))
        LB_rows = np.zeros((self.L,))

        UB_cols = np.zeros((self.M,))
        LB_cols = np.zeros((self.M,))

        eps = 0.1

        while t <= self.T:

            l += 1
            t_l[1] = np.ceil(16 * np.log(self.T) / (delta_tilde ** 2))


            active_rows = (np.arange(self.L) == h_u).astype(int)  # [1,0,0,1...] 1 iff active arm in row
            active_cols = (np.arange(self.M) == h_v).astype(int)

            # Exploration
            for k in range(int(t_l[1] - t_l[0])):
                t_l[0] = t_l[1]

                # Explore rows
                chosen_col = np.random.randint(0, self.M)
                j = h_v[chosen_col]
                for i in np.argwhere(active_rows == 1).flatten():
                    if t <= self.T:
                        row_reward[i, j] = row_reward[i, j] + self.draw_arm((i, j), t, cumrewards_per_arm, mu_hat,
                                                                            return_rew=True)
                        t += 1
                    else:
                        break

                # Explore cols
                chosen_row = np.random.randint(0, self.L)
                i = h_u[chosen_row]

                for j in np.argwhere(active_cols == 1).flatten():
                    if t <= self.T:
                        col_reward[i, j] = col_reward[i, j] + self.draw_arm((i, j), t, cumrewards_per_arm, mu_hat,
                                                                            return_rew=True)
                        t += 1
                    else:
                        break

            # Compute Upper and lowerbounds
            delta_eps = (1 + eps) * np.log(self.T)  # + 3*np.log(np.log(self.T)) # ?????

            for i in np.argwhere(active_rows == 1):
                u_hat = np.sum(row_reward[i, :]) / t_l[1]
                UB_rows[i] = tools.klUCB_rank1(t_l=t_l[1], delta_eps=delta_eps, u_hat=u_hat)
                LB_rows[i] = tools.klLCB_rank1(t_l=t_l[1], delta_eps=delta_eps, u_hat=u_hat)

            for j in np.argwhere(active_cols == 1):
                v_hat = np.sum(col_reward[:, j]) / t_l[1]
                UB_cols[j] = tools.klUCB_rank1(t_l=t_l[1], delta_eps=delta_eps, u_hat=v_hat)
                LB_cols[j] = tools.klLCB_rank1(t_l=t_l[1], delta_eps=delta_eps, u_hat=v_hat)

            # Elimination
            best_LB_row_idx = tools.random_argmax(LB_rows)
            for i in range(self.L):
                if UB_rows[h_u[i]] <= LB_rows[best_LB_row_idx]:
                    h_u[i] = best_LB_row_idx

            best_LB_col_idx = tools.random_argmax(LB_cols)
            for j in range(self.M):
                if UB_cols[h_v[j]] <= LB_cols[best_LB_col_idx]:
                    h_v[j] = best_LB_col_idx

            # Update delta_tilde
            delta_tilde /= 1.5

        print("Last active rows:",np.argwhere(active_rows == 1).flatten())
        print("Last active cols:",np.argwhere(active_cols == 1).flatten())
        self.is_updated = True

    def plot_regret(self):
        assert self.is_updated == True
        plt.plot(self.regret_hist, label="Rank1ELimKL regret")

    def plot_img_arm_drawn(self):
        assert self.is_updated == True, "algo non updated "
        self.nb_times_drawn = self.compute_nb_times_drawn()
        nb_times_drawn_reshaped = self.nb_times_drawn.reshape(self.matrix_uv.shape)

        plt.imshow(np.log(nb_times_drawn_reshaped))
        plt.colorbar()
