
import sys
##sys.path.append("/home/cindy/Documents/memoire/code_git/")
sys.path.append("../")
import numpy as np
from Arms.Arm import PairArm

class Environment:
    def __init__(self, list_of_arms):
        self.list_of_arms = list_of_arms
        for idx, arm in enumerate(list_of_arms):
            if not arm.idx: arm.idx = idx
        self.nb_arms = len(list_of_arms)

        self.opt_arm = self.find_opt_arm()


    def find_opt_arm(self):
        opt_arm = self.list_of_arms[0]
        max_mu = opt_arm.mu
        for cur_arm in self.list_of_arms:
            if cur_arm.mu > max_mu:
                max_mu = cur_arm.mu
                opt_arm = cur_arm
        return opt_arm


    def get_arm_idx(self, idx):
    	# for cur_arm in list_of_arms:
    	# 	if cur_arm.idx == idx:
    	# 		arm = cur_arm
        try:
            return self.list_of_arms[idx]
        except:
            return "Arm " + str(idx) + " not found"
