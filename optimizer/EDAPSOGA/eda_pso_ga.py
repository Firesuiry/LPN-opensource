import numpy as np
from eva_func import Func
from benchmark import test_fit
from optimizer.base import optimizer

DIM = 50
N_PART = 256
from functools import partial


def sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-arr))


def evaluate_fun(arr: np.ndarray):
    return np.sum(arr, axis=1)


class EDAPSOGA(optimizer):

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams=hyperparams)
        self.name = 'EDAPSOGA'
        self.n_part = self.npart
        self.n_dim = self.ndim
        self.random_choice = self.championship_choice
        self.fun = fun
        assert self.n_part % 2 == 0, '粒子数必须是偶数'

        self.cross_over_rate = 0.8 * self.n_part // 1
        self.mutation_rate = 0.005 * self.n_part // 1
        self.re_generate_rate = 0.15 * self.n_part // 1
        self.pso_rate = 0.8 * self.n_part // 1
        # print(self.__dict__)

        assert self.cross_over_rate + self.re_generate_rate + 2 <= self.n_part, '各项产生的种群数不能大于种群总个数'

        self.xs = np.random.randint(0, 2, (self.n_part, self.n_dim), dtype=np.bool)
        self.vs = np.random.random((self.n_part, self.n_dim)) * 8 - 4

        self.prob = np.zeros_like(self.xs[0], dtype=np.int)
        self.fits = np.zeros((self.n_part,), dtype=np.float)

        self.atom_history_best_x = self.xs.copy()
        self.atom_history_best_fit = np.zeros((self.n_part,), dtype=np.int)
        self.history_best_x = self.xs[0].copy()
        self.history_best_fit = 0

        # self.data_store = []

    def caculate_prob(self, eda_m):
        self.prob = np.sum(eda_m, axis=0)

    def choice(self):
        eda_m = []
        for i in range(self.n_part // 2):
            if self.fits[i] > self.fits[i + 1]:
                eda_m.append(self.xs[i].copy())
            else:
                eda_m.append(self.xs[i + 1].copy())
        self.caculate_prob(eda_m)

    def update_best(self):
        best_index = np.argmax(self.fits)
        if self.fits[best_index] > self.history_best_fit:
            self.history_best_x = self.xs[best_index].copy()
            self.history_best_fit = self.fits[best_index]

        for i in range(self.n_part):
            if self.fits[i] > self.atom_history_best_fit[i]:
                self.atom_history_best_x[i] = self.xs[i].copy()
                self.atom_history_best_fit[i] = self.fits[i]

    def pso(self):
        best_index = np.argmax(self.fits)
        if self.fits[best_index] > self.history_best_fit:
            self.history_best_x = self.xs[best_index].copy()
            self.history_best_fit = self.fits[best_index]

        for i in range(self.n_part):
            if self.fits[i] > self.atom_history_best_fit[i]:
                self.atom_history_best_x[i] = self.xs[i].copy()
                self.atom_history_best_fit[i] = self.fits[i]

        for i in range(2, self.n_part):
            self.vs[i] = 0.6 * self.vs[i] + 2 * np.random.random() * (
                    self.atom_history_best_x[i] * 1 - self.xs[i] * 1) + \
                         2 * np.random.random() * (self.history_best_x * 1 - self.xs[i] * 1)
            rnd = np.random.randint(0, self.n_part)
            if rnd < self.pso_rate:
                self.xs[i] = np.random.random(self.n_dim) > sigmoid(self.vs[i])

    def mutation(self):
        for i in range(self.n_part):
            rnd = np.random.randint(0, self.n_part)
            if rnd <= self.mutation_rate:
                index = np.random.randint(0, self.n_dim)
                self.xs[i][index] = np.abs(1 - self.xs[i][index])

    def championship_choice(self, member=2):
        indexs = np.random.randint(0, self.n_part, (member,))
        max_index = np.argmax(self.fits[indexs])
        return indexs[max_index]

    def roulette_choice(self):
        rnd = np.random.random()
        roulette_rate = self.fits / (np.sum(self.fits) + 1e-307)
        for i in range(self.n_part):
            rnd -= roulette_rate[i]
            if rnd <= 0:
                return i
        return i

    def reproduct(self):
        next_gen = np.zeros_like(self.xs)

        # 留下两个精英
        if self.fits[0] > self.fits[1]:
            elite_index = np.array((0, 1))
        else:
            elite_index = np.array((1, 0))

        for i in range(2, self.n_part):
            if self.fits[i] > self.fits[elite_index[1]]:
                elite_index[1] = i
                if self.fits[i] > self.fits[elite_index[0]]:
                    elite_index[1] = elite_index[0]
                    elite_index[0] = i

        next_gen[0] = self.xs[elite_index[0]]
        next_gen[1] = self.xs[elite_index[1]]

        # 其余的随机交叉
        for i in range(2, self.n_part):
            rnd = np.random.randint(0, self.n_part)
            target1 = self.random_choice()
            if rnd < self.cross_over_rate:
                # 交叉操作
                target2 = self.random_choice()
                crossover_point = np.random.randint(0, self.n_dim)
                new_gen = self.xs[target1].copy()
                new_gen_v = self.vs[target1].copy()
                new_gen[crossover_point:] = self.xs[target2][crossover_point:]
                new_gen_v[crossover_point:] = self.vs[target2][crossover_point:]
                next_gen[i] = new_gen
                self.vs[i] = new_gen_v
            elif rnd < self.cross_over_rate + self.re_generate_rate:
                # 根据分布生成
                next_gen[i] = np.random.randint(0, self.n_part, self.n_dim) > self.prob
            else:
                # 随机保留一个后代
                next_gen[i] = self.xs[i].copy()

        self.xs = next_gen

    def run(self):
        while self.fes < self.max_fes:
            self.choice()
            self.reproduct()
            self.mutation()
            self.pso()
            self.fits = self.evaluate(self.xs)
            self.update_best()
            # data = {
            #     'max': np.max(self.fits),
            #     'mean': np.mean(self.fits),
            #     'std': np.std(self.fits),
            # }
            # self.data_store.append(data)
        # return np.max(self.fits), self.data_store

    def get_best_value(self):
        return self.history_best_fit

