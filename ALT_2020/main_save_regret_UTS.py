import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import os


from Games import utils_draws as ud
from Games import UnimodalGame as ug

from Environments import Rank1Env as r1e
from Policies import OSUB
from Policies import UTS
import time


if __name__ == "__main__":
    ## Parameters
    draws_dir = "./draws"
    results_dir = "./results"
    algo_name = "UTS"
    list_draw_leader_every = [2]#2,5,10,15,30]


    ## FILL if single, Else, list_test = none
    horizon = 300000
    nb_row =  16
    nb_col =  nb_row
    two_different_arms = [0.75,0.25]
    prefix_name = f'two_diff_{two_different_arms[0]}-{two_different_arms[1]}'
    list_test = [f'{prefix_name}_row-{nb_row}_col-{nb_col}_horizon-{horizon}']
    nb_pkl=200
    ##

    if list_test:
        list_draw_name = list_test
    else:
        list_draw_name = [fn for fn in os.listdir(draws_dir) if (os.path.isdir(os.path.join(draws_dir, fn)))]

    for cur_draw_name in list_draw_name:
        horizon = int(cur_draw_name.split('horizon-')[1])
        nb_row = int(cur_draw_name.split('row-')[1].split('_')[0])
        nb_col = int(cur_draw_name.split('col-')[1].split('_')[0])
        cur_draw_folder = os.path.join(draws_dir, cur_draw_name)
        list_pkl = [fn for fn in os.listdir(cur_draw_folder) if fn.endswith('p')]
        list_pkl = sorted(list_pkl,key=lambda x: int(x.split(".")[0]))
        if nb_pkl==None:
            nb_pkl = len(list_pkl)

        cur_env_results_dir = os.path.join(results_dir, cur_draw_name, algo_name)
        if not os.path.exists(cur_env_results_dir):
            os.mkdir(cur_env_results_dir)


        print(list_draw_leader_every)
        for i_dle, draw_leader_every in enumerate(list_draw_leader_every):
            print(f'cur_draw_name {cur_draw_name}, draw_leader_every {draw_leader_every}')
            out_dle_dir = os.path.join(cur_env_results_dir, "dle_" + str(draw_leader_every))
            if not os.path.exists(out_dle_dir):
                os.mkdir(out_dle_dir)
            print(list_pkl[:nb_pkl], len(list_pkl[:nb_pkl]))
            for i_pkl, cur_pkl in enumerate(list_pkl[:nb_pkl]):
                print(cur_pkl)
                out_regret_filename = os.path.join(out_dle_dir, "regret_" + cur_pkl)
                if not os.path.exists(out_regret_filename):
                    tic = time.time()
                    with open(os.path.join(cur_draw_folder, cur_pkl), 'rb') as f:
                        draws_dict = p.load(f)
                    draws_in_advance = draws_dict["draws_in_advance"]
                    assert (len(draws_dict['mu_row']), len(draws_dict['mu_col']))  == (nb_row, nb_col)


                    ## UTS for current dle, cur pkl
                    UTS_rank1_env = r1e.create_rank1env(draws_dict)
                    UTS_policy = UTS.UTS(draw_leader_every=draw_leader_every)
                    UTS_game = ug.UnimodalGame(environment=UTS_rank1_env, policy=UTS_policy, horizon=horizon,save_arm_drawn=False)
                    UTS_game.playGame()


                    with  open(out_regret_filename, 'wb') as f:
                        p.dump(UTS_game.regret_history,f)

                    toc = time.time() - tic
                    tps_restant_sec = toc*(nb_pkl-i_pkl) + (len(list_draw_leader_every) - i_dle - 1)*nb_pkl*toc
                    print(f"Pkl-{i_pkl}, dle-{draw_leader_every}, toc-{toc}, Temps restant estimé: {tps_restant_sec/60} minutes")
                    del draws_dict
                    del draws_in_advance
                    del UTS_rank1_env
                    del UTS_policy
                    del UTS_game
