import numpy as np


def run(config, fun):
    # creates model
    fes = config.get('fes')
    dim = config.get('dim')

    best_fit = 0
    best_sol = None
    for fe in range(fes//int(1e6)):
        # if fe * 1e8 % fes == 0:
        #     print('fe:{}'.format(fe))
        xs = np.random.randint(0, 2, (int(1e6), dim), dtype=np.int)
        scores = fun(xs)
        index = np.argmax(scores)
        score = scores[index]
        x = xs[index]
        if score > best_fit:
            best_fit = score
            best_sol = x

    print(f'维度：{dim},最优答案：{best_fit} sol:{best_sol} ')

    return {}
