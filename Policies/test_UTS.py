import unittest

import numpy as np
import sys
sys.path.append("../")
import numpy as np
from Policies.OSUB import OSUB
from Policies.UTS import UTS
import Environments.Rank1Env as r1e
import pickle as p
from Arms import Arm
import tools

class TestPolicy(unittest.TestCase):
    def test_UTS(self):
        mu_row = np.linspace(0.1, 0.5, 5)
        mu_col = np.linspace(0.1, 0.3, 3)
        with open('../test_folder/pair_draw_1000.p', 'rb') as f:
            draws_dict = p.load(f)
        draws_in_advance = draws_dict['draws_in_advance']
        my_rank1_env = r1e.create_rank1env(mu_row, mu_col, draws_in_advance)
