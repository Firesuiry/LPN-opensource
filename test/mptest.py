from multiprocessing.dummy import freeze_support

import numpy as np
import multiprocessing as mp


def excuter(a):
    print(f'excuter {a}')
    return a + 3


def runner(a):
    configs = [i for i in range(a, a + 10, 1)]
    print(configs)
    pool1 = mp.Pool(processes=2)
    res = pool1.map(excuter, configs)
    print(f'runner a:{a} res:{res}')
    return res


if __name__ == '__main__':
    freeze_support()
    pool2 = mp.Pool(processes=2)
    ass = [(i * 10) for i in range(0, 10, 1)]
    ress = pool2.map(runner, ass)
    print(ress)
