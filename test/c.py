import ctypes
import numpy as np
import time
from tqdm import tqdm
from logger import logger
import multiprocessing as mp

print('c.py start')
# lib = ctypes.cdll.LoadLibrary("./fwht.dll")
lib = ctypes.cdll.LoadLibrary("C:\\Users\\cass\\source\\repos\\fwht\\x64\\Release\\fwht.dll")
lib.FWHT.restype = ctypes.c_uint64

exp2 = [2 ** i for i in range(128)]


def fwht(data, n):
    size = ctypes.c_longlong(2 ** n)
    s = time.time()
    pure_s = time.time()
    pos = lib.FWHT(data, size, n)
    pure_use_time = time.time() - pure_s
    logger.info(f'fwht use_time:{time.time() - s}|{pure_use_time}')

    return pos


class Func:

    def __init__(self, dim, error_rate=0., fixed_seed=False, prob_1=0.5, row=2 ** 22):
        self.dim = dim
        self.row = int(row)
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
        logger.debug(f'right sol:{self.x.T},best score:{self.best_value}')

    def evaluate(self, xs: np.ndarray):
        if len(xs.shape) == 1:
            assert xs.shape[0] == self.dim
        else:
            assert xs.shape[1] == self.dim
        res = np.matmul(self.a, xs.T) % 2
        score = self.row - np.sum(np.abs(res - self.b_f), axis=0)
        return np.array(score, dtype=float)

    def majority_counter(self):
        logger.debug('start mc')
        long_array_class = ctypes.c_long * 2 ** self.dim
        mc = long_array_class()
        # mc = np.zeros((2 ** self.dim,), dtype=int)
        exp2_ls = np.array([2 ** (self.dim - d - 1) for d in range(self.dim)])
        for i in range(self.row):
            if i % 2 ** 20 == 0:
                logger.debug(f'mc {i // 2 ** 20}/{self.row // 2 ** 20}')
            # index = 0
            # for d in range(self.dim):
            #     index += self.a[i, d] * exp2[self.dim - d - 1]
            index = np.sum(np.multiply(self.a[i,], exp2_ls))
            # assert index == index2
            if self.b_f[i] > 0:
                mc[index] -= 1
            else:
                mc[index] += 1
        return mc

    def wht_solve(self):
        mc = self.majority_counter()
        del self.a
        pos = fwht(mc, self.dim)
        right_x = self.x.reshape(-1)
        guess_x = hex2bin(pos, right_x.shape[0])
        # logger.debug(right_x)
        # logger.debug(guess_x)
        error = np.sum(np.abs(guess_x - right_x)) > 0
        return error


def test_solve(run_times=10):
    logger.info(f'test solve :{run_times}')
    t = time.time()
    error_times = 0
    for i in tqdm(range(run_times)):
        single_t = time.time()
        f = Func(26, error_rate=0.49)
        error = f.wht_solve()
        if error:
            error_times += 1
        logger.info(f'one epoch done, use time:{time.time() - single_t},error:{error}')
    logger.info(f'use time :{(time.time() - t) / run_times} error rate:{error_times / run_times}')


def hex2bin(int_hex, num):
    ls = []
    while int_hex > 0:
        ls.append(int_hex % 2)
        int_hex = int_hex // 2
    while len(ls) < num:
        ls.append(0)
    ls.reverse()
    ls = np.array(ls)
    return ls


def run(config):
    dim = config.get('dim', 24)
    row = config.get('row', 2 ** 20)
    f = Func(dim, error_rate=0.499, row=row)
    error = f.wht_solve()
    return error


def log2file(info, file):
    logger.info(info)
    with open(file, 'a+') as f:
        f.write(info)
        f.write('\n')


if __name__ == '__main__':
    # data = [1, 2, 1, 1, 3, 2, 1, 2]
    # data = list(range(2 ** 25))
    # data1 = fwht(data, 25)
    # for i in range(20, 30):
    #     print(f'dim:{i}')
    #     f = Func(i, error_rate=0.499)
    #     error = f.wht_solve()
    # errors = []
    max_run = 128

    # for i in range(max_run):
    #     f = Func(24, error_rate=0.499)
    #     error = f.wht_solve()
    #     print(error)
    #     errors.append(error)
    # print(errors)
    # print(np.sum(errors) / max_run)
    log2file(f'start run {time.time()}',
             'logs/scrate.txt')
    pool = mp.Pool(processes=4)
    dim = 28
    for row in [2 ** row for row in range(20, 26)]:
        config = {
            'row': row,
            'dim': dim,
        }
        errors = pool.map(run, [config] * max_run)

        print(errors)
        print(np.sum(errors) / max_run)
        log2file(f'dim:{dim} log2row:{np.log2(row)} error_rate:{0.499} error:{np.sum(errors) / max_run * 100}% all:{max_run}',
                 'logs/scrate.txt')

    # data2 = fwht(data1)
    # print(data)
    # print(data1)
    # print(data2)
    # test_solve()
    # print(hex2bin(8))
