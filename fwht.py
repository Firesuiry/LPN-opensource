import numpy as np
import time
from tqdm import tqdm


def fwht(data):
    print('fwht')
    data = data.copy()
    n = int(np.log2(len(data)))
    for i in tqdm(range(0, n, 1)):
        for j in range(0, 2 ** n, 2 ** (i + 1)):
            for k in range(0, 2 ** i, 1):
                a = j + k
                b = j + k + 2 ** i
                # print(f'i:{i} j:{j} k:{k} a:{a} b:{b}')
                tmp = data[a]
                data[a] += data[b]
                data[b] = tmp - data[b]
    return data


def fwht_statistic(n):
    a = 0
    for i in range(0, n, 1):
        for j in range(0, 2 ** n, 2 ** (i + 1)):
            # for k in range(0, 2 ** i, 1):
            a += 2 * 2 ** i
    print(f'n:{n} num:{a} 理论：{n*2**n} delta:{a-n*2**n}')
    return a


class Func:

    def __init__(self, dim, error_rate=0., fixed_seed=False, prob_1=0.5):
        self.dim = dim
        self.row = 2 ** 20
        self.lie = dim
        # Ax=b
        if fixed_seed:
            np.random.seed(1)
        # self.a = np.random.randint(0, 2, (self.row, dim), dtype=np.int)
        self.a = np.array(np.random.random((self.row, dim)) < prob_1, dtype=int)
        self.x = np.random.randint(0, 2, (dim, 1), dtype=int)
        self.b = np.array(np.matmul(self.a, self.x) % 2, dtype=int)
        self.b_f = np.array(np.random.random(self.b.shape) < error_rate, dtype=int)
        self.b_f = np.array(np.abs(self.b_f - self.b), dtype=int)
        self.best_value = self.evaluate(self.x.T)
        print(f'right sol:{self.x.T},best score:{self.best_value}')

    def evaluate(self, xs: np.ndarray):
        if len(xs.shape) == 1:
            assert xs.shape[0] == self.dim
        else:
            assert xs.shape[1] == self.dim
        res = np.matmul(self.a, xs.T) % 2
        score = self.row - np.sum(np.abs(res - self.b_f), axis=0)
        return np.array(score, dtype=float)

    def majority_counter(self):
        print('majority_counter')
        mc = np.zeros((2 ** self.dim,), dtype=int)
        for i in tqdm(range(self.row)):
            index = 0
            for d in range(self.dim):
                index += self.a[i, d] * 2 ** (self.dim - d - 1)
            if self.b[i] > 0:
                mc[index] -= 1
            else:
                mc[index] += 1
        return mc

    def wht_solve(self):
        mc = self.majority_counter().tolist()
        mc = fwht(mc)
        pos = np.argmax(mc)
        print(f'{bin(pos)}')
        print(self.x)
        # index = 0
        # for d in range(self.dim):
        #     index += self.x[d] * 2 ** d
        # print(index)


def test_solve():
    f = Func(20, error_rate=0.49)
    t = time.time()
    f.wht_solve()
    print(f'use time :{time.time() - t}')


if __name__ == '__main__':
    # data = [1, 2, 1, 1, 3, 2, 1, 2]
    # data = [0, 1, 2, 3]
    # data1 = fwht(data)
    # print(data1)
    # data2 = fwht(data1)
    # print(data)
    # print(data1)
    # print(data2)
    # test_solve()
    for i in range(5,30):
        fwht_statistic(i)
