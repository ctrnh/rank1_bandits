import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.Policy import Policy
from Environments.UnimodalEnvironment import UnimodalEnvironment
from tools import tools

class UTS_frac(Policy):
	def __init__(self,
	 			#draw_leader_every,
				x_gamma,
				y_gamma):
		#self.draw_leader_every = draw_leader_every
		self.isUnimodalPolicy = True
		self.name = "UTS_gamma_frac"
		self.x_gamma = x_gamma
		self.y_gamma = y_gamma
		self.list_dle = [i*self.y_gamma//self.x_gamma for i in range(self.x_gamma)]
		assert self.x_gamma <= self.y_gamma
		assert len(self.list_dle) == self.x_gamma

	def playArm(self,
				env,
				t):
		#print("list_dle = ", self.list_dle)
		assert isinstance(env, UnimodalEnvironment), "not unimodal environment for unimodal policy"
		
		if t < env.nb_arms:
			arm_t = env.get_arm_idx(t)
			leader_t = None
		else:
			list_arm_mu_hat = [(arm, arm.mu_hat) for arm in env.list_of_arms]
			leader_t = tools.best_arm(list_arm_mu_hat)

			if leader_t.dle_alpha in self.list_dle:
				#print("draw leader", leader_t.dle_alpha)
				arm_t = leader_t
			else:
				if not leader_t.neighbors:
					env.set_neighbors(leader_t)

				theta_neighbors = []
				for cur_neighbor_arm in leader_t.neighbors:
					theta_neighbors.append((cur_neighbor_arm, self.draw_theta(cur_neighbor_arm)))
				arm_t = tools.best_arm(theta_neighbors)

			leader_t.dle_alpha += 1
			if leader_t.dle_alpha == self.y_gamma:
				leader_t.dle_alpha = 0


		reward_t = arm_t.draw(t)

		return arm_t, reward_t, leader_t

	def draw_theta(self, arm):
		a = arm.cumreward + 1
		b = arm.nb_times_drawn - arm.cumreward + 1
		return np.random.beta(a, b)
