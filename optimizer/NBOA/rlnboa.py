import functools

from benchmark import test_fit
from env.TestpsoEnv import TestpsoEnv
from eva_func import Func
from optimizer.NBOA.nboa import NBOA
from rl.DDPG.TF2_DDPG_Basic import DDPG
import numpy as np
import re

MODEL_PATH = R'D:\develop\probcaculate\rl\train0\ddpg_actor_episode100.h5'


class RLNBOA(NBOA):

    def __init__(self, maxfe, fun, npart, ndim, model_file=MODEL_PATH, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams=hyperparams)
        model_names = re.split('\W+', model_file)
        self.name = 'RLNBOA-{}-{}'.format(model_names[-3], model_names[-2])
        gym_env = TestpsoEnv(show=False)
        ddpg = DDPG(gym_env, discrete=False, memory_cap=1000000, gamma=0, sigma=0.25, actor_units=(16,),
                    critic_units=(8, 16, 32), use_priority=False, lr_critic=1e-8, lr_actor=1e-10)
        ddpg.load_actor(model_file)
        self.actor = ddpg

    def run(self):
        while self.fes < self.max_fes:
            state = self.get_state()
            action = self.actor.policy(state)
            self.run_single(all_action=action)


if __name__ == '__main__':
    dim = 25
    fes = 2 ** dim // 8
    # fit = functools.partial(test_fit, func_num=0)
    ress = []
    for i in range(10):
        f = Func(dim)
        test_fun = f.evaluate
        best_value = f.best_value
        nboa = RLNBOA(fes, test_fun, 100, dim)
        nboa.run()
        res = {
            'detact': nboa.get_best_value(),
            'target': best_value,
        }
        ress.append(res)
    print(ress)
