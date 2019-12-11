import sys
##sys.path.append("/home/cindy/Documents/memoire/code_git/")
sys.path.append("../")
import numpy as np
from Arms.Arm import PairArm
from Environments.UnimodalEnvironment import UnimodalEnvironment



def create_rank1env(draws_dict):
    """
    draws_in_advance = list of nb_arms lists of length horizon
    """
    draws_in_advance = draws_dict["draws_in_advance"]
    mu_row =  draws_dict["mu_row"]
    mu_col =  draws_dict["mu_col"]
    nb_row = len(mu_row)
    nb_col = len(mu_col)

    mu_matrix = np.dot(mu_row.reshape(nb_row,1), mu_col.reshape(1, nb_col))
    mu_flat = mu_matrix.flatten()

    list_of_pair_arms = []
    for cur_idx_flat, cur_mu in enumerate(mu_flat):
        cur_idx_pair = np.unravel_index(cur_idx_flat, mu_matrix.shape)
        cur_draws_in_advance = draws_in_advance[cur_idx_flat]
        cur_arm = PairArm(mu=cur_mu,
                          draws_in_advance=cur_draws_in_advance)
        cur_arm.idx = cur_idx_flat
        cur_arm.set_idx_pair(cur_idx_pair)
        list_of_pair_arms.append(cur_arm)

    rank1env = Rank1Env(mu_row = mu_row,
                        mu_col = mu_col,
                        mu_matrix = mu_matrix,
                        mu_flat = mu_flat,
                        list_of_pair_arms = list_of_pair_arms)
    return rank1env

class Rank1Env(UnimodalEnvironment):
    def __init__(self,
                 mu_row,
                 mu_col,
                 mu_matrix,
                 mu_flat,
                 list_of_pair_arms):
        self.mu_row = mu_row # list of row means ("u")
        self.nb_row = len(mu_row)
        self.mu_col = mu_col # "v"
        self.nb_col = len(mu_col)

        self.mu_matrix = mu_matrix
        self.mu_flat = mu_flat

        super().__init__(list_of_pair_arms)

        ## Create self.matrix_arms?


    def get_arm_idx(self, idx):
        if type(idx) == tuple:
            idx = np.ravel_multi_index(idx, self.mu_matrix.shape)
        return self.list_of_arms[idx]




    def set_neighbors(self, arm,
                    arm_included=True):
        list_of_neighbors = set()

        (arm_row, arm_col) = arm.idx_pair

        for row in range(self.nb_row):
            if row != arm_row:
                list_of_neighbors.add(self.get_arm_idx(idx=(row, arm_col)))

        for col in range(self.nb_col):
            if col != arm_col:
                list_of_neighbors.add(self.get_arm_idx(idx=(arm_row, col)))

        if arm_included:
            list_of_neighbors.add(arm)


        arm.neighbors = list_of_neighbors
