import functools
import random

from benchmark import test_fit
from eva_func import Func
from optimizer.base import optimizer

import numpy as np

ALL_RANDOM_RATE = 0.01
EDA_RANDOM_RATE = 0.02
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.2
BPSO_RATE = 0.6
CLPSO_RATE = 0.5
FINE_TUNE_ONE = 0.2
FINE_TUNE_TWO = 0.1
CLPSO_FLAG_M = 7


class NBOA(optimizer):

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams=hyperparams)
        self.name = 'nboa'

        self.xs = np.random.randint(0, 2, (npart, ndim), dtype=np.int)
        self.fits = np.zeros(self.npart)

        # bpso
        self.pbest = self.xs.copy()
        self.pbest_fit = self.fits.copy()
        self.pbest_fit[:] = 0

        self.gbest = self.xs[0].copy()
        self.gbest_fit = self.fits[0].copy()
        self.gbest_fit = 0

        self.v1s = np.random.random((self.npart, self.ndim)) * 4 - 2
        self.v0s = np.random.random((self.npart, self.ndim)) * 4 - 2

        # eda
        self.prob = np.zeros_like(self.xs[0], dtype=np.int)
        self.prob[:] = 0.5

        # clpso
        self.flag = np.zeros(self.npart)
        self.fid = np.zeros((self.npart, self.ndim), dtype=np.int)
        indexs = np.array(list(range(self.npart)))
        self.atom_pci = 0.05 + 0.45 * np.exp(10 * indexs / (self.npart - 1)) / (np.exp(10) - 1)

        self.best_change_fe = 0

        for i in range(self.npart):
            self.caculate_fid(i)

    def run(self):
        while self.fes < self.max_fes:
            self.run_single()

    # ALL_RANDOM_RATE = 0.01
    # EDA_RANDOM_RATE = 0.02
    # MUTATION_RATE = 0.02
    # CROSSOVER_RATE = 0.2
    # BPSO_RATE = 0.6
    # FINE_TUNE_ONE = 0.2
    # FINE_TUNE_TWO = 0.1
    def run_single(self,
                   all_action=None
                   ):
        oldfe = self.fes
        for i in range(self.npart):
            if all_action is not None:
                actions = all_action[i % 5 * 10:i % 5 * 10 + 10]
                all_random_rate = (actions[0] + 1) * 0.01
                eda_random_rate = (actions[1] + 1) * 0.01
                mutation_rate = (actions[2] + 1) * 0.01
                crossover_rate = (actions[3] + 1) * 0.1
                bpso_rate = (actions[4] + 1) * 0.5
                clpso_rate = (actions[5] + 1) * 0.5
                fine_tune_one = (actions[6] + 1) * 0.5
                fine_tune_two = (actions[7] + 1) * 0.2
                self.run_single_particle(i,
                                         all_random_rate=all_random_rate,
                                         eda_random_rate=eda_random_rate,
                                         mutation_rate=mutation_rate,
                                         crossover_rate=crossover_rate,
                                         bpso_rate=bpso_rate,
                                         clpso_rate=clpso_rate,
                                         fine_tune_one=fine_tune_one,
                                         fine_tune_two=fine_tune_two,
                                         )
            else:
                self.run_single_particle(i)
        self.update_best()
        use_fe = (self.fes - oldfe) / self.max_fes
        return use_fe, self.fes / self.max_fes

    def run_single_particle(self,
                            particle_id,
                            all_random_rate=ALL_RANDOM_RATE,
                            eda_random_rate=EDA_RANDOM_RATE,
                            mutation_rate=MUTATION_RATE,
                            crossover_rate=CROSSOVER_RATE,
                            bpso_rate=BPSO_RATE,
                            clpso_rate=CLPSO_RATE,
                            fine_tune_one=FINE_TUNE_ONE,
                            fine_tune_two=FINE_TUNE_TWO
                            ):

        if self.flag[particle_id] >= CLPSO_FLAG_M:
            self.caculate_fid(particle_id)
            self.flag[particle_id] = 0
        no_update = (1 + 10 * (self.fes - self.best_change_fe) / self.max_fes)
        params = all_random_rate * self.flag[
            particle_id] * no_update, eda_random_rate, mutation_rate, crossover_rate, bpso_rate
        params = np.array(params)
        params = params / np.sum(params)
        actions = self.all_random, self.eda_random, self.mutation, self.crossover, functools.partial(self.bpso,
                                                                                                     clpso=clpso_rate)
        action = np.random.choice(actions, p=params)
        action(particle_id)
        fine_tune_r = np.random.random()
        if fine_tune_r < fine_tune_one:
            if fine_tune_r < fine_tune_one * fine_tune_two:
                self.fine_tune(particle_id, 2)
            else:
                self.fine_tune(particle_id, 1)

    def mutation(self, particle_id):
        index = np.random.randint(0, self.ndim)
        self.xs[particle_id][index] = np.abs(1 - self.xs[particle_id][index])

    def crossover(self, particle_id, rate=0.2):
        crossover_indexs = np.random.random(self.ndim) < 0.2
        target = self.random_choice()
        self.xs[particle_id][crossover_indexs] = self.xs[target][crossover_indexs]

    def random_choice(self):
        rnd = np.random.random()
        roulette_rate = self.fits / (np.sum(self.fits) + 1e-307)
        for i in range(self.npart):
            rnd -= roulette_rate[i]
            if rnd <= 0 or i == (self.npart - 1):
                return i

    def bpso(self, particle_id, weight=0.89, clpso=0.5, c1=2, c2=2):
        i = particle_id
        cr1 = c1 * np.random.random(self.ndim)
        cr2 = c2 * np.random.random(self.ndim)

        r = np.random.random()
        if r < clpso:
            # clpso
            target = np.zeros(self.ndim)
            for d in range(self.ndim):
                target[d] = self.xs[self.fid[i, d]][d]
            d11 = (target - 0.5) * 2 * cr1
            d01 = -d11
            d12 = 0
            d02 = -d12
        else:

            d11 = (self.pbest[i] - 0.5) * 2 * cr1
            d01 = -d11
            d12 = (self.gbest - 0.5) * 2 * cr2
            d02 = -d12

        new_v1 = weight * self.v1s[i] + d11 + d12
        self.v1s[i] = new_v1
        new_v0 = weight * self.v0s[i] + d01 + d02
        self.v0s[i] = new_v0
        old_x = self.xs[i]
        select_v = np.abs(self.xs[i]) * self.v1s[i] + np.abs(self.xs[i] - 1) * self.v0s[i]
        sigmoid_select_v = sigmoid(select_v)

        fanzhuan = 1 - np.array(np.random.random(sigmoid_select_v.shape) > sigmoid_select_v, np.bool) * 2
        new_x = fanzhuan * (self.xs[i] - 0.5) + 0.5
        self.xs[i] = new_x

    def all_random(self, particle_id):
        self.xs[particle_id] = np.random.randint(0, 2, self.ndim, dtype=np.int)

        self.v1s[particle_id] = np.random.random(self.ndim) * 4 - 2
        self.v0s[particle_id] = np.random.random(self.ndim) * 4 - 2

        self.flag[particle_id] = 0

    def eda_random(self, particle_id):
        self.xs[particle_id] = np.random.random(self.ndim) < self.prob

        self.v1s[particle_id] = np.random.random(self.ndim) * 4 - 2
        self.v0s[particle_id] = np.random.random(self.ndim) * 4 - 2

        self.flag[particle_id] = 0

    def caculate_fid(self, i):
        #  Selection of exemplar dimensions for particle i.
        rands = np.random.uniform(0, 1, self.ndim)
        for d in range(self.ndim):
            rand = rands[d]
            if rand < self.atom_pci[i]:
                fids = np.random.randint(0, self.npart, 2)
                fits = self.pbest_fit[fids]
                self.fid[i, d] = fids[np.argmax(fits)]
            else:
                self.fid[i, d] = i

    def fine_tune(self, particle_id, level=1):
        new_x = self.xs[particle_id].copy()
        new_fit = self.fits[particle_id].copy()
        better = False
        if level == 1:
            # 1 阶finetune
            for d in range(self.ndim):
                test_x = self.xs[particle_id].copy()
                test_x[d] = np.abs(1 - test_x[d])
                fit = self.evaluate(test_x)
                if fit > new_fit:
                    new_fit = fit
                    new_x = test_x
                    better = True
        else:
            # 2 阶finetune
            for d1 in range(self.ndim):
                for d2 in range(d1, self.ndim, 1):
                    test_x = self.xs[particle_id].copy()
                    test_x[d1] = np.abs(1 - test_x[d1])
                    test_x[d2] = np.abs(1 - test_x[d2])
                    fit = self.evaluate(test_x)
                    if fit > new_fit:
                        new_fit = fit
                        new_x = test_x
                        better = True
        self.xs[particle_id] = new_x
        self.fits[particle_id] = new_fit
        if better:
            self.fine_tune(particle_id, level)

    def update_best(self):
        self.fits = self.evaluate(self.xs)
        for i in range(self.npart):
            if self.better(self.fits[i], self.pbest_fit[i]):
                self.pbest_fit[i] = self.fits[i].copy()
                self.pbest[i] = self.xs[i].copy()
                self.flag[i] = 0

                if self.better(self.fits[i], self.gbest_fit):
                    self.gbest_fit = int(self.fits[i].copy())
                    self.gbest = self.xs[i].copy()
                    self.best_change_fe = self.fes
                    # print(f'best change: sol:{self.gbest} val:{self.gbest_fit} fe:{self.fes * 100 // self.max_fes}')
            else:
                self.flag[i] += 1

        # eda 生成
        eda_m = []
        for i in range(self.npart // 2):
            if self.fits[i] > self.fits[i + 1]:
                eda_m.append(self.xs[i].copy())
            else:
                eda_m.append(self.xs[i + 1].copy())
        self.prob = np.sum(eda_m, axis=0) * 0.3 + 0.7 * self.prob

    def get_state(self):
        return np.array((self.fes / self.max_fes, self.best_change_fe / self.max_fes)) * 2 - 1

    def data_store(self):
        self.record[self.fes] = {
            'best': int(self.gbest_fit),
        }

    def get_best_value(self):
        return self.gbest_fit


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    dim = 25
    fes = 2 ** dim // 10
    fit = functools.partial(test_fit, func_num=0)
    test_fun = Func(dim).evaluate
    nboa = NBOA(fes, test_fun, 100, dim)
    nboa.run()
