import unittest

import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.OSUB import OSUB
import Environments.Rank1Env as r1e
import pickle as p
from Arms import Arm
import tools

class TestPolicy(unittest.TestCase):
    def test_OSUB(self):
        mu_row = np.linspace(0.1, 0.5, 5)
        mu_col = np.linspace(0.1, 0.3, 3)
        with open('../test_folder/pair_draw_1000.p', 'rb') as f:
            draws_dict = p.load(f)
        draws_in_advance = draws_dict['draws_in_advance']
        my_rank1_env = r1e.create_rank1env(mu_row, mu_col, draws_in_advance)

        my_policy = OSUB(draw_leader_every=3)
        for t in range(my_rank1_env.nb_arms):
            arm_t, reward_t, leader_t = my_policy.playArm(my_rank1_env, t)
            self.assertIsInstance(arm_t, Arm.PairArm)
            self.assertEqual(arm_t.idx, t)
            self.assertTrue(reward_t == 0 or reward_t == 1)


        for t in range(my_rank1_env.nb_arms , 500):
            arm_t, reward_t, leader_t = my_policy.playArm(my_rank1_env, t)
            self.assertIsInstance(arm_t, Arm.PairArm)
            self.assertIsInstance(leader_t, Arm.PairArm)
            self.assertTrue(reward_t == 0 or reward_t == 1)

if __name__ == '__main__':
    unittest.main()
