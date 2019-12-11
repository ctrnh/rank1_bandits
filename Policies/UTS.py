import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.Policy import Policy
from Environments.UnimodalEnvironment import UnimodalEnvironment
from tools import tools

class UTS(Policy):
	def __init__(self, draw_leader_every):
		self.draw_leader_every = draw_leader_every
		self.isUnimodalPolicy = True
		self.name = "UTS"

	def playArm(self,
				env,
				t):
		assert isinstance(env, UnimodalEnvironment), "not unimodal environment for unimodal policy"
		nb_arms = env.nb_arms

		if t < nb_arms:
			arm_t = env.get_arm_idx(t)
			leader_t = None
		else:
			list_arm_mu_hat = [(arm, arm.mu_hat) for arm in env.list_of_arms]
			leader_t = tools.best_arm(list_arm_mu_hat)
			leader_t.nb_times_leader += 1

			if self.draw_leader_every!="no_leader" and leader_t.nb_times_leader%self.draw_leader_every == 0:
				#print(f'Leader_t = {leader_t.idx} has been leader for {leader_t.nb_times_leader}, {leader_t.nb_times_leader//self.draw_leader_every}')
				arm_t = leader_t

			else:
				if not leader_t.neighbors:
					env.set_neighbors(leader_t)

				theta_neighbors = []
				for cur_neighbor_arm in leader_t.neighbors:
					theta_neighbors.append((cur_neighbor_arm, self.draw_theta(cur_neighbor_arm)))
				arm_t = tools.best_arm(theta_neighbors)


		reward_t = arm_t.draw(t)
		#print(f'draw arm {arm_t.idx}')

		return arm_t, reward_t, leader_t

	def draw_theta(self, arm):
		a = arm.cumreward + 1
		b = arm.nb_times_drawn - arm.cumreward + 1
		return np.random.beta(a, b)
