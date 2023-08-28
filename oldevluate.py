from functools import partial
from benchmark import test_fit
from optimizer.EDAPSOGA.eda_pso_ga import Agent
import numpy as np


def evaluate_single(agent, func_num, run_num=10):
    print(f'evaluate_single fit:{func_num} run_num:{run_num}')
    vmaxs = []
    for _ in range(run_num):
        fit = partial(test_fit, func_num=func_num)
        a = agent(fit)
        Vmax, data = a.run()
        vmaxs.append(Vmax)
    print(f'fit:{func_num},res:{np.mean(vmaxs)}')


if __name__ == '__main__':
    for i in range(3):
        evaluate_single(Agent, i)
