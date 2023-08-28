from env.EnvBase import Env
import numpy as np

import random

from eva_func import Func
from optimizer.NBOA.nboa import NBOA

DIM = 50


class TestpsoEnv(Env):
    def __init__(self, show=False, dim=25):
        super().__init__(obs_shape=(2,), action_shape=(50,), action_low=-1, action_high=1)
        self.optimizer = None
        self.fit_value = [0., 0., 0., 0., 0.]

        self.step_num = 0
        self.show_flag = show

        self.fun_num = -1
        self.min_value = 0

        self.run_time = 0
        self.best_value = np.inf

        self.dim = dim

    def reset(self):
        """

        :return: next_state
        """
        n_dim = self.dim
        self.n_run = n_run = 1000
        n_part = (n_dim // 5) * 20
        show = self.show_flag

        func = Func(n_dim, fixed_seed=False)
        fun = func.evaluate
        self.best_value = func.best_value
        self.optimizer = NBOA(2 ** n_dim // 10, fun, n_part, n_dim)

        self.fit_value = [0., 0., 0., 0., 0.]
        self.step_num = 0
        self.old_data = {
            'mean': 0,
            'best': 0,
        }

        # next_state, reword, done, _ = self.step(None, init=True)
        return self.optimizer.get_state()

    def test(self):
        done = False
        step_num = 0
        self.reset()
        while not done:
            a, b, done, c = self.step(None, True)
            step_num += 1
        return step_num

    def step(self, action, init=False):
        """
        :param action: 动作
        :return:
        next_state
        reword
        done
        none
        """

        action = action.numpy()
        done = False
        self.step_num += 1

        if self.optimizer.show:
            self.optimizer.show_method()

        usefe,fes = self.optimizer.run_single(action)

        # if self.pso_swarm.best_fit < self.fun.finish or self.step_num >= self.n_run:
        #     done = True
        if not self.optimizer.run_flag:
            done = True

        old_best = self.old_data['best']
        self.old_data['best'] = self.optimizer.gbest_fit

        deta_best = self.optimizer.gbest_fit - old_best

        next_state = self.optimizer.get_state()
        # next_state.append(self.step_num * 0.001)
        # print(f'state:{next_state}\naction:{action[:10]}\nmean:{np.mean(action)},std:{np.std(action)}')

        if deta_best != 0:
            # reward = sqrt(deta_best, 3)
            reward = 1 * (0.1 + 2 * fes)
        else:
            reward = -10 * usefe

        if np.isnan(reward):
            print(deta_best)
            assert 0

        if self.optimizer.gbest_fit == self.best_value:
            reward = 10
            done = True

        if init:
            reward = 0
        if self.show_flag:
            print('action:{} next_state:{} reward:{} done:{} best:{}'.format(action, next_state, reward, done,
                                                                             self.optimizer.history_best_fit))
        if done:
            res = f'迭代次数：{self.step_num},运行结果：{self.optimizer.gbest_fit}'
            print(res)
            if self.show_flag:
                print('迭代次数：{}'.format(self.step_num))
            with open('res2.txt', 'a', encoding='utf-8') as f:
                f.write(f'{res}\n')
        return np.array(next_state), reward, done, None
