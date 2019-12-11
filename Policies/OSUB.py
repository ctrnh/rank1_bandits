import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.Policy import Policy
from Environments.UnimodalEnvironment import UnimodalEnvironment
from tools import tools

class OSUB(Policy):
	def __init__(self, draw_leader_every):
		self.draw_leader_every = draw_leader_every
		self.isUnimodalPolicy = True
		self.name = "OSUB"

	def playArm(self,
				env,
				#mu_hat_history,
				t):
		assert isinstance(env, UnimodalEnvironment), "not unimodal environments"
		nb_arms = env.nb_arms

		if t < nb_arms:
			arm_t = env.get_arm_idx(t)
			leader_t = None
		else:
			list_arm_mu_hat = [(arm, arm.mu_hat) for arm in env.list_of_arms]
			leader_t = tools.best_arm(list_arm_mu_hat)
			leader_t.nb_times_leader += 1

			if self.draw_leader_every!='no_leader' and leader_t.nb_times_leader%self.draw_leader_every == 0:
				arm_t = leader_t

			else:
				if not leader_t.neighbors:
					env.set_neighbors(leader_t)

				UCB_neighbors_idx = []
				for cur_neighbor_arm in leader_t.neighbors:
					UCB_neighbors_idx.append((cur_neighbor_arm, self.KLUCB_idx(cur_neighbor_arm, t)))
				arm_t = tools.best_arm(UCB_neighbors_idx)
				


		reward_t = arm_t.draw(t)

		return arm_t, reward_t, leader_t


	def UCB1_idx(self,
				  arm,
				  t):
		return arm.mu_hat + np.sqrt((2*np.log(t))/arm.nb_times_drawn)

	def KLUCB_idx(self, arm, t):
		lb = arm.mu_hat# arm.mu_hat
		ub = 1
		c = 3 # a changer ?
		d = (np.log(t) + c*np.log(np.log(t)))/ arm.nb_times_drawn
		return tools.klucb(lb, d, tools.klBern, ub)
