from optimizer.base import optimizer
import numpy as np


class RS(optimizer):

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams=hyperparams)
        self.name = 'RS'

        self.best_fit = 0
        self.best_sol = None

    def run(self):
        while self.fes < self.max_fes:
            xs = np.random.randint(0, 2, (int(2**10), self.ndim), dtype=np.int)
            scores = self.evaluate(xs)
            index = np.argmax(scores)
            score = scores[index]
            x = xs[index]
            if score > self.best_fit:
                self.best_fit = score
                self.best_sol = x
        self.data_store()

        # print(f'维度：{dim},最优答案：{best_fit} sol:{best_sol} ')

    def data_store(self):
        self.record[self.fes] = {
            'best': self.best_fit,
        }

    def get_best_value(self):
        return self.best_fit
