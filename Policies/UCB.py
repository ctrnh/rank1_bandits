import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.Policy import Policy
from tools import tools

class UCB(Policy):
	def __init__(self):
		self.name = "UCB"

	def playArm(self,
				env,
				t):
		nb_arms = env.nb_arms

		if t < nb_arms:
			arm_t = env.get_arm_idx(t)
		else:
			list_arm_UCB = [(arm, self.UCB_idx(arm, t)) for arm in env.list_of_arms]
			arm_t = tools.best_arm(list_arm_UCB)

		reward_t = arm_t.draw(t)

		return arm_t, reward_t


	def UCB_idx(self,
				  arm,
				  t):
		return arm.mu_hat + np.sqrt((2*np.log(t))/arm.nb_times_drawn)

class UCB1(UCB):
	def __init__(self):
		self.name = "UCB1"

	def UCB_idx(self,
				  arm,
				  t):
		return arm.mu_hat + np.sqrt((2*np.log(t))/arm.nb_times_drawn)

class KLUCB(UCB):
	def __init__(self):
		self.name = "KL-UCB"

	def UCB_idx(self,
				  arm,
				  t):
		lb = arm.mu_hat
		ub = 1
		c = 3 # a changer ?
		d = (np.log(t) + c*np.log(np.log(t))) / arm.nb_times_drawn
		return tools.klucb(lb, d, tools.klBern, ub)
