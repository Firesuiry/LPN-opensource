from optimizer.disABC import Hive
from optimizer.disABC.BeeBinary import BeeBinary
import numpy as np
# 优化目标为最小化
def run(config, fun):
    # creates model
    fes = config.get('fes')
    ndim = config.get('dim')
    bee_prototype = BeeBinary(dimensions=ndim,
                              fun=fun)

    model = Hive.BeeHive(bee_prototype=bee_prototype,
                         numb_bees=50,
                         max_itrs=200,
                         maxfes=fes)

    # runs model
    model.run()
    print("Solution ABC: {0}".format(model.best.vector))
    print("Fitness Value ABC: {0}".format(fun(np.array(model.best.vector))))

    assert model.best is not None
    return model
