import time

from bernuli import CmbinationNumber
from logger import logger

import numpy as np

ERROR_RATE = 0.35


class Func:

    def __init__(self, dim, error_rate=ERROR_RATE, fixed_seed=False, prob_1=0.5, row=-1):
        self.dim = dim
        if row > 0:
            self.row = row
        else:
            self.row = 2 ** dim
        self.lie = dim
        # Ax=b
        if fixed_seed:
            np.random.seed(1)
        # self.a = np.random.randint(0, 2, (self.row, dim), dtype=np.int)
        self.a = np.array(np.random.random((self.row, dim)) < prob_1, dtype=int)
        self.x = np.random.randint(0, 2, (dim,), dtype=int)
        self.b = np.array(np.matmul(self.a, self.x) % 2, dtype=int).reshape(-1, 1)
        self.b_f = np.array(np.random.random(self.b.shape) < error_rate, dtype=int)
        self.b_f = np.array(np.abs(self.b_f - self.b), dtype=int)
        self.best_value = self.evaluate(self.x)
        print(f'right sol:{self.x.T},best score:{self.best_value}')

    def evaluate(self, xs: np.ndarray):
        if len(xs.shape) == 1:
            xs = xs.reshape(1, -1).copy()
        assert xs.shape[1] == self.dim
        s = time.time()
        res = np.matmul(self.a, xs.T) % 2
        punish = np.sum((xs > 0.5) * (1 - xs) ** 2 + (xs <= 0.5) * xs ** 2)
        score = 1 - np.sum(np.abs(res - self.b_f), axis=0) / self.row
        logger.debug(f'caculate xs.shape:{xs.shape}, use time:{time.time() - s}')
        return np.array(score, dtype=float)

    def get_some_sample(self, hwmin, hwmax, row_max=2000):
        s = time.time()
        new_a = []
        new_b = []
        new_bf = []
        prob1 = []

        a_sum = np.sum(self.a, axis=1) / self.dim
        indexs = (a_sum <= hwmax) * (hwmin <= a_sum)
        new_a = self.a[indexs]
        new_b = self.b[indexs]
        new_bf = self.b_f[indexs]
        prob1 = a_sum[indexs]

        # for i in range(self.row):
        #     if hwmax >= np.sum(self.a[i]) / self.dim >= hwmin:
        #         new_a.append(self.a[i].copy())
        #         new_b.append(self.b[i])
        #         new_bf.append(self.b_f[i])
        #         prob1.append(np.sum(self.a[i]) / self.dim)
        logger.info(f'采样任务 范围{hwmin}-{hwmax} 采样前总数:{self.row} '
                    f'采样后总数:{len(new_a)} 平均HW：{np.mean(prob1)} '
                    f'用时:{time.time() - s}')
        if len(new_a) > row_max:
            iter_a, iter_b, iter_bf = [], [], []
            if hwmin <= 0:
                iter_a, iter_b, iter_bf = self.get_some_sample(hwmin, hwmax - 1 / self.dim, row_max=row_max)
            elif hwmax >= 1:
                iter_a, iter_b, iter_bf = self.get_some_sample(hwmin + 1 / self.dim, hwmax, row_max=row_max)
            if len(iter_a) > 2000:
                return iter_a, iter_b, iter_bf
        return np.array(new_a), np.array(new_b).reshape((-1, 1)), np.array(new_bf).reshape((-1, 1))

    def caculate_result(self, a, bf, xs):
        if len(xs.shape) == 1:
            xs = xs.reshape(1, -1).copy()
        assert xs.shape[1] == self.dim
        s = time.time()
        res = np.matmul(a, xs.T) % 2
        score = 1 - np.sum(np.abs(res - bf), axis=0) / a.shape[0]
        logger.debug(f'caculate xs.shape:{xs.shape}, use time:{time.time() - s}')
        return np.array(score, dtype=np.float)


class LowProb1Func(Func):
    def __init__(self, dim, error_rate=ERROR_RATE, fixed_seed=False, prob_1_max=0.5, row=-1):
        self.dim = dim
        if row > 0:
            self.row = row
        else:
            self.row = 2 ** dim
        self.lie = dim
        # Ax=b
        if fixed_seed:
            np.random.seed(1)
        distribution = []
        for i in range(int(prob_1_max * dim) + 1):
            distribution.append(CmbinationNumber(dim, i))
        distribution = distribution / np.sum(distribution)
        prob1s = np.random.choice(list(range(int(prob_1_max * dim) + 1)), self.row, p=distribution)
        self.a = np.zeros((self.row, self.dim))
        for i in range(self.row):
            change_indexs = np.random.choice(list(range(self.dim)), prob1s[i], replace=False)
            self.a[i][change_indexs] = 1

        # self.a = np.random.randint(0, 2, (self.row, dim), dtype=np.int)
        # self.a = np.array(np.random.random((self.row, dim)) < prob_1, dtype=int)
        self.x = np.random.randint(0, 2, (dim,), dtype=int)
        self.b = np.array(np.matmul(self.a, self.x) % 2, dtype=int).reshape(-1, 1)
        self.b_f = np.array(np.random.random(self.b.shape) < error_rate, dtype=int)
        self.b_f = np.array(np.abs(self.b_f - self.b), dtype=int)


def test_dim_error_rate(dim, error_rate):
    f = Func(dim, error_rate)
    xs = np.random.randint(0, 2, (999999, dim), dtype=np.int)
    xs[0] = f.x.T
    score = f.evaluate(xs)
    print(f'维度：{dim} 错误率：{error_rate}，正确答案：{score[0]},最优答案：{np.max(score[1:])}')
    if score[0] != np.max(score):
        exit()


if __name__ == '__main__':
    f = LowProb1Func(dim=20, row=10000)
    print(f.evaluate(f.x))
