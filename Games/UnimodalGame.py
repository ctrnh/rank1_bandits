import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from Games.Game import Game
from Environments.UnimodalEnvironment import UnimodalEnvironment
import os
import pickle as p
from tqdm import tqdm

class UnimodalGame(Game):
    def __init__(self,
                environment,
                policy,
                horizon,
                save_arm_drawn):

        super().__init__(environment,
                        policy,
                        horizon,
                        save_arm_drawn)

        assert isinstance(self.env, UnimodalEnvironment)
        assert self.policy.isUnimodalPolicy

        self.leader_history = []


    def playGame(self, save_regret_every=1):
        regret_t = 0
        for t in tqdm(range(self.horizon)):
            arm_t, reward_t, leader_t = self.policy.playArm(self.env,
                                                            t)
            regret_t += self.opt_arm.mu - arm_t.mu

            if self.save_arm_drawn:
                self.arm_drawn_history.append(arm_t.idx)
                if leader_t is not None:
                    self.leader_history.append(leader_t.idx)
                else:
                    self.leader_history.append(-1)
            if t%save_regret_every==0:
                self.regret_history.append(regret_t)




    def plot_and_save(self,
                      output_dir,
                      show_regret=True,
                      show_arm=True,
                      show_mu_hat=True,
                      show_leader=True,
                      save_game=True):
        if show_regret:
            plt.figure()
            plt.plot(self.regret_history)
            plt.title("Regret history")
            plt.xlabel("t")
            plt.ylabel("Regret")
            plt.savefig(os.path.join(output_dir,f"{self.policy.name}_regret.jpg"))
        if show_arm:
            plt.figure()
            plt.subplot(121)
            idx_arm_drawn_history = [cur_arm.idx for cur_arm in self.arm_drawn_history]
            plt.plot(np.arange(self.horizon), idx_arm_drawn_history, 'r*')
            plt.title("Arm drawn history")
            plt.xlabel("t")
            plt.ylabel("idx arm drawn")

            plt.subplot(122)
            mu_arm_drawn_history = [cur_arm.mu for cur_arm in self.arm_drawn_history]
            plt.plot(np.arange(self.horizon), mu_arm_drawn_history, 'r*')
            plt.title("Arm drawn history")
            plt.xlabel("t")
            plt.ylabel("mu arm drawn")

            plt.savefig(os.path.join(output_dir,f"{self.policy.name}_arm_drawn_history.jpg"))
        if show_mu_hat:
            plt.figure()
            plt.subplot(121)
            mu_hat_matrix = np.zeros((self.env.nb_arms, self.horizon))
            for i, cur_arm in enumerate(self.env.list_of_arms):
                mu_hat_matrix[i,:] = cur_arm.mu_hat_history + [0 for i in range(len(cur_arm.mu_hat_history), self.horizon)]
            plt.imshow(mu_hat_matrix,interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.title("Mu hat history")

            plt.subplot(122)
            mu_matrix = np.zeros((self.env.nb_arms, self.horizon))
            for i, cur_arm in enumerate(self.env.list_of_arms):
                mu_hat_matrix[i,:] = cur_arm.mu
            plt.imshow(mu_hat_matrix,interpolation='nearest', aspect='auto')

            plt.title("Mu ")
            plt.colorbar()
            plt.savefig(os.path.join(output_dir,f"{self.policy.name}mu_hat_history.jpg"))

        if show_leader:
            plt.figure()
            idx_leader_history = [cur_leader.idx for cur_leader in self.leader_history if cur_leader is not None]
            plt.plot(idx_leader_history, 'b*')
            plt.title("Leader history")
            plt.xlabel("t")
            plt.ylabel("Leader")
            plt.savefig(os.path.join(output_dir,f"{self.policy.name}_leader.jpg"))

        if save_game:
            with open(os.path.join(output_dir,f"{self.policy.name}_game.p"), 'wb') as f:
                p.dump(self, f)
