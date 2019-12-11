import numpy as np
class Arm:
    def __init__(self,
                mu,
                draws_in_advance=None,
                law="Bernoulli"):
        self.idx = None # index is set only when we define the environment
        self.mu = mu
        self.law = law
        self.draws_in_advance = draws_in_advance
        self.mu_hat = 0
        self.cumreward = 0
        self.mu_hat_history = []
        self.nb_times_drawn = 0

        self.nb_times_leader = 0
        self.dle_alpha = 0
        self.neighbors = None
        assert 0 <= self.mu <= 1

    def draw(self, t):
        if self.draws_in_advance is not None:
            assert t < self.draws_in_advance.shape[0]
            reward = self.draws_in_advance[t]

        elif self.law == 'Bernoulli':
            reward = np.random.binomial(1, self.mu)

        self.mu_hat = (self.mu_hat * self.nb_times_drawn + reward) / (self.nb_times_drawn + 1)

        while len(self.mu_hat_history) < t :
            if self.nb_times_drawn > 0:
                self.mu_hat_history.append(self.mu_hat_history[-1])
            else:
                self.mu_hat_history.append(None)

        self.mu_hat_history.append(self.mu_hat)
        self.nb_times_drawn += 1
        self.cumreward += reward
        assert len(self.mu_hat_history) == t + 1, f'arm {self.idx}, t={t}, mu_hat_history length = {len(self.mu_hat_history)}'

        return reward

    def set_idx(self,idx):
        self.idx = idx




class PairArm(Arm):
    def __init__(self,
                mu,
                draws_in_advance=None,
                law="Bernoulli"):
        super().__init__(mu=mu,
                         draws_in_advance=draws_in_advance,
                         law=law)

        self.idx_pair = None
        self.idx_row = None
        self.idx_col = None

        #useful for Rank1ElimKL
        self.upper_bound = None
        self.lower_bound = None
        self.active = True



    def set_idx_pair(self,idx_pair):
        self.idx_pair = idx_pair
        self.idx_row = idx_pair[0]
        self.idx_col = idx_pair[1]
