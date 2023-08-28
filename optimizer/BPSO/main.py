from optimizer.BPSO.bpso import bpso


def run(config, fun):
    # creates model
    fes = config.get('fes')
    ndim = config.get('dim')

    model = bpso(fes, fun, 100, ndim)

    # runs model
    model.run()
    #
    # print("Fitness Value ABC: {0}".format(model.best.fitness))
    # print("Solution ABC: {0}".format(model.best.vector))
    return model