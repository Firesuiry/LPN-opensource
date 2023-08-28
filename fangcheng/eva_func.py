import numpy as np

ERROR_RATE = 0.35

class Func:

    def __init__(self, dim, error_rate=ERROR_RATE, fixed_seed=False):
        self.dim = dim
        self.row = np.int(1000)
        self.lie = dim
        # Ax=b
        if fixed_seed:
            np.random.seed(1)
        self.a = np.random.randint(0, 2, (self.row, dim), dtype=np.int)
        self.x = np.random.randint(0, 2, (dim, 1), dtype=np.int)
        self.b = np.array(np.matmul(self.a, self.x) % 2, dtype=np.int)
        self.b_f = np.array(np.random.random(self.b.shape) < error_rate, dtype=np.int)
        self.b_f = np.array(np.abs(self.b_f - self.b), dtype=np.int)
        self.best_value = self.evaluate(self.x.T)
        print(f'right sol:{self.x.T},best score:{self.best_value}')

    def evaluate(self, xs: np.ndarray):
        if len(xs.shape) == 1:
            assert xs.shape[0] == self.dim
        else:
            assert xs.shape[1] == self.dim
        res = np.matmul(self.a, xs.T) % 2
        score = self.row - np.sum(np.abs(res - self.b_f), axis=0)
        return np.array(score, dtype=np.float)


if __name__ == '__main__':
    f = Func(20)
    f.evaluate(f.x.T)
