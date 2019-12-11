import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.Policy import Policy
from Environments.Environment import Environment
from tools import tools

class TS(Policy):
	def __init__(self):
		self.name = "TS"

	def playArm(self,
				env,
				t):
		nb_arms = env.nb_arms

		if t < nb_arms:
			arm_t = env.get_arm_idx(t)
		else:
			list_arm_theta = [(arm, self.draw_theta(arm)) for arm in env.list_of_arms]
			arm_t = tools.best_arm(list_arm_theta)


		reward_t = arm_t.draw(t)

		return arm_t, reward_t, leader_t


	def draw_theta(self,
				  arm):
        a = arm.cumreward + 1
        b = self.nb_times_drawn - arm.cumreward + 1
		return np.random.beta(a, b)
