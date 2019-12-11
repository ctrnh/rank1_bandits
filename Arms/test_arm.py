import unittest
import Arm
import pickle as p

class TestArm(unittest.TestCase):
    def test_idx_arm(self):
        my_arm = Arm.Arm(mu=0)
        my_arm.set_idx(3)
        self.assertEqual(my_arm.idx,3)

    def test_draw_not_advance(self):
        mu = 0.3
        my_arm = Arm.Arm(mu=mu)
        nb_times_first_draws = 5000
        for t in range(nb_times_first_draws):
            reward_t = my_arm.draw(t)
            assert reward_t == 0 or reward_t == 1
        assert len(my_arm.mu_hat_history) == nb_times_first_draws
        assert my_arm.nb_times_drawn == nb_times_first_draws
        assert abs(my_arm.mu_hat - mu) < 0.02

        second_t = nb_times_first_draws + 100
        my_arm.draw(second_t)
        assert my_arm.nb_times_drawn == nb_times_first_draws + 1
        for i in range(100):
            assert my_arm.mu_hat_history[nb_times_first_draws - 1 + i] == my_arm.mu_hat_history[nb_times_first_draws - 1]

    def test_draw_advance(self):
        with open('../test_folder/single_draw_1000.p', 'rb') as f:
            draws_dict = p.load(f)
        my_arm = Arm.Arm(mu=0,
                         draws_in_advance=draws_dict['draws_in_advance'][0])
        my_arm.idx = 0
        cur_t = 0
        reward = my_arm.draw(cur_t)
        assert reward == 0 or reward == 1


    def test_pair_arm(self):
        idx_pair = (0, 0)
        idx = 0
        with open('../test_folder/pair_draw_1000.p', 'rb') as f:
            draws_dict = p.load(f)
        my_arm = Arm.PairArm(mu=0,
                             draws_in_advance=draws_dict['draws_in_advance'])
        my_arm.set_idx_pair(idx_pair)
        my_arm.draw(0)




if __name__ == '__main__':
    unittest.main()
