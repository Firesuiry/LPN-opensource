import numpy as np

from optimizer.base import optimizer


def sigmf(x):
    return 1 / (1 + np.exp(-10 * (x - 0.5)))


class bgwopso(optimizer):
    # optimizer target is minimize
    def __init__(self, maxfe, fun, npart, ndim,hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim,hyperparams=hyperparams)
        self.name = 'BGWOPSO'
        self.alpha_pos = np.zeros((1, ndim))
        self.alpha_score = 0

        self.beta_pos = np.zeros((1, ndim))
        self.beta_score = 0

        self.delta_pos = np.zeros((1, ndim))
        self.delta_score = 0

        self.xs = np.random.randint(0, 2, (npart, ndim), dtype=np.int)
        self.vs = np.random.random((npart, ndim)) * 0.3

        self.w = 0.5 + np.random.random() / 2

        self.fits = np.zeros(self.npart)

    def run(self):
        while self.fes < self.max_fes:
            # print(f'sol:{self.alpha_pos},score:{self.alpha_score}')
            self.run_single()
        # print(f'sol:{self.alpha_pos},score:{self.alpha_score}')

        return self.alpha_pos, self.alpha_score

    def update_best(self):
        for i in range(self.npart):
            fitness = self.fits[i]
            if fitness > self.alpha_score:
                # print('alpha update')
                self.alpha_score = fitness
                self.alpha_pos = self.xs[i].copy()

            elif self.alpha_score > fitness > self.beta_score:
                # print('beta update')
                self.beta_score = fitness
                self.beta_pos = self.xs[i].copy()

            elif self.alpha_score > fitness > self.delta_score and fitness < self.beta_score:
                # print('delta update')
                self.delta_score = fitness
                self.delta_pos = self.xs[i].copy()

    def run_single(self):
        self.fits = self.evaluate(self.xs)
        self.update_best()
        for i in range(self.npart):
            a = np.ones(self.ndim) * 2 * (1 - self.fes / self.max_fes)
            r1 = np.random.random(self.ndim)
            a1 = 2 * a * r1 - a
            c1 = 0.5
            d_alpha = abs(c1 * self.alpha_pos - self.w * self.xs[i])
            v1 = sigmf(-a1 * d_alpha) > np.random.random(self.ndim)
            x1 = (v1 + self.alpha_pos) > 1

            r1 = np.random.random(self.ndim)
            a2 = 2 * a * r1 - a
            c2 = 0.5
            d_beta = abs(c2 * self.beta_pos - self.w * self.xs[i])
            v2 = sigmf(-a2 * d_beta) > np.random.random(self.ndim)
            x2 = (v2 + self.beta_pos) > 1

            r1 = np.random.random(self.ndim)
            a3 = 2 * a * r1 - a
            c3 = 0.5
            d_delta = abs(c3 * self.delta_pos - self.w * self.xs[i])
            v3 = sigmf(-a3 * d_delta) > np.random.random(self.ndim)
            x3 = (v3 + self.delta_pos) > 1

            r1 = np.random.random(self.ndim)
            r2 = np.random.random(self.ndim)
            r3 = np.random.random(self.ndim)
            self.vs[i] = self.w * (self.vs[i] +
                                   c1 * r1 * (x1 - self.xs[i]) +
                                   c2 * r2 * (x2 - self.xs[i]) +
                                   c3 * r3 * (x3 - self.xs[i]))
            new_x = (sigmf((x1 + x2 + x3) / 3) + self.vs[i]) > np.random.random(self.ndim)
            self.xs[i] = new_x

    def data_store(self):
        self.record[self.fes] = {
            'best': self.alpha_score,
        }

    def get_best_value(self):
        return self.alpha_score
