import logging

import numpy as np

from eva_func import Func
from logger import logger
import multiprocessing as mp
# logger.setLevel(logging.DEBUG)
import time
import pandas as pd


class BruteSearch:

    def __init__(self, k, once_k):
        self.k = k
        self.once_k = once_k
        self.max = 2 ** (k - once_k)
        self.iter_num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_num < self.max:
            res = np.zeros((2 ** self.once_k, self.k), dtype=int)
            # 前 k-once_k位赋值
            remainder = self.iter_num
            index = 0
            while remainder > 0:
                new_r = remainder // 2
                res[:, self.k - self.once_k - index - 1] = remainder - 2 * new_r
                remainder = new_r
                index += 1
            # 后面赋值
            for i in range(2 ** self.once_k):
                remainder = i
                index = 0
                while remainder > 0:
                    new_r = remainder // 2
                    res[i, self.k - index - 1] = remainder - 2 * new_r
                    remainder = new_r
                    index += 1
            self.iter_num += 1
            return res
        else:
            raise StopIteration()


def check_need_row(config):
    error_rate = config['error_rate']
    k = config['k']
    weight = config['weight']
    row_min = 10
    row_max = 10
    logger.info(f'step 1')
    while row_max == row_min:
        logger.info(f'min:{row_min},max:{row_max}')
        test_row = row_min * 2
        if check_success(test_row, error_rate, k, weight):
            row_max = test_row
        else:
            row_max = row_min = test_row

    logger.info(f'step 2')
    while row_max - row_min > 10:
        test_row = int((row_min + row_max) / 2)
        logger.info(f'min:{row_min},max:{row_max} test_row:{test_row}')
        if check_success(test_row, error_rate, k, weight):
            row_max = test_row
        else:
            row_min = test_row
    data = {
        'row': int((row_max + row_min) / 2),
        'error_rate': error_rate,
        'k': k,
        'weight': weight
    }
    return data


def check_success(row, error_rate, k, weight):
    all_run = 10
    success = all_run
    for i in range(all_run):
        if success < all_run:
            break
        f = Func(dim=k, error_rate=error_rate, prob_1=weight, row=row)
        while f.best_value < 0.99 - error_rate:
            f = Func(dim=k, error_rate=error_rate, prob_1=weight, row=row)
        x_generate = BruteSearch(k=k, once_k=10)
        for xs in x_generate:
            res = f.evaluate(xs)
            if np.max(res) > f.best_value:
                index = np.argmax(res)
                logger.info(f'find better best res:{res[index]} x:{xs[index]}')
                success -= 1
                break
    logger.info(f'check_success:{success > 9} row:{row},error_rate:{error_rate} k:{k} weight:{weight}')
    return success > 9


if __name__ == '__main__':
    configs = []
    pool = mp.Pool(processes=2)
    for error_rate in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        for k in [10, 11, 12, 13, 14, 15]:
            for weight in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                configs.append({
                    'error_rate': error_rate,
                    'k': k,
                    'weight': weight
                })
    ress = pool.map(check_need_row, configs[:10])
    df = pd.DataFrame(ress)
    df.to_csv('result.csv', index=False)
