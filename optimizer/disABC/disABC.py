from eva_func import Func
from optimizer.base import optimizer
from optimizer.disABC import Hive
from optimizer.disABC.BeeBinary import BeeBinary
import numpy as np


class disABC(optimizer):

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        super().__init__(maxfe, fun, npart, ndim, hyperparams)
        self.name = 'disABC'

        self.bee_prototype = BeeBinary(dimensions=ndim,
                                       fun=self.evaluate)

        self.model = Hive.BeeHive(bee_prototype=self.bee_prototype,
                                  numb_bees=50,
                                  max_itrs=200,
                                  maxfes=maxfe)

    def run(self):
        self.model.run()

    def data_store(self):
        self.record[self.fes] = {
            'best': self.fun(np.array(self.model.best.vector)),
        }


if __name__ == '__main__':
    func = Func(10)
    test_fun = func.evaluate
    a = disABC(1e5, test_fun, 40, 10)
    a.run()
