from optimizer.BGWOPSO.bgwopso import bgwopso
from optimizer.disABC import Hive
from optimizer.disABC.BeeBinary import BeeBinary


def run(config, fun):
    # creates model
    fes = config.get('fes')
    ndim = config.get('dim')
    def fun2(*args,**kwargs):
        return -fun(*args,**kwargs)
    model = bgwopso(fes, fun2, 100, ndim)

    # runs model
    model.run()
    #
    # print("Fitness Value ABC: {0}".format(model.best.fitness))
    # print("Solution ABC: {0}".format(model.best.vector))
    return model
