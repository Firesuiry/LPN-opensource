from optimizer.base import optimizer
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class bpso(optimizer):

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams)
        self.name = 'BPSO'

        self.xs = np.random.randint(0, 2, (npart, ndim), dtype=np.int)
        self.fits = np.zeros(self.npart)

        self.pbest = self.xs.copy()
        self.pbest_fit = self.fits.copy()
        self.pbest_fit[:] = -np.inf

        self.gbest = self.xs[0].copy()
        self.gbest_fit = self.fits[0].copy()
        self.gbest_fit = -np.inf

        self.v1s = np.random.random((self.npart, self.ndim)) * 4 - 2
        self.v0s = np.random.random((self.npart, self.ndim)) * 4 - 2

        self.c1 = self.c2 = 2

    def run_single(self):
        weight = 1 - self.fes / self.max_fes

        for i in range(self.npart):
            cr1 = self.c1 * np.random.random(self.ndim)
            cr2 = self.c2 * np.random.random(self.ndim)
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

        self.update_best()


    def run(self):
        while self.fes < self.max_fes:
            self.run_single()

    def update_best(self):
        self.fits = self.evaluate(self.xs)
        for i in range(self.npart):
            if self.better(self.fits[i], self.pbest_fit[i]):
                self.pbest_fit[i] = self.fits[i].copy()
                self.pbest[i] = self.xs[i].copy()

                if self.better(self.fits[i], self.gbest_fit):
                    self.gbest_fit = self.fits[i].copy()
                    self.gbest = self.xs[i].copy()
        # print(f'best change: sol:{self.gbest} val:{self.gbest_fit}')

    def data_store(self):
        self.record[self.fes] = {
            'best': self.gbest_fit,
        }
