from functools import partial

from benchmark import test_fit
from eva_func import Func
from optimizer.disABC.main import run as disABC_run
from optimizer.BGWOPSO.main import run as bgwopso_run
from optimizer.random_search.main import run as random_run
from optimizer.BPSO.main import run as bpso_run
from config import setting as config

test_fun = Func(config.get('dim'))
fit = partial(test_fit, func_num=0)
bpso_run(config, test_fun.evaluate)
# disABC_run(config, fit)


''''
Fitness Value ABC: 0.0021645021645021645
Solution ABC: 
[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
0 0 1 0 1 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0
random 维度：20,最优答案：678 sol:[0 0 1 0 1 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0]
'''