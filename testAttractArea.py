import numpy as np
import matplotlib.pyplot as plt
from eva_func import Func
import multiprocessing as mp
from psolpn import near_vector


# def near_vector(x: np.ndarray, radius: int, start_at: int):
#     xs = []
#     if radius < 1:
#         return [x]
#     for d in range(start_at, x.shape[0], 1):
#         new_x = x.copy()
#         new_x[d] = 1 - new_x[d]
#         if radius > 1:
#             xs += near_vector(new_x, radius - 1, d + 1)
#         else:
#             xs.append(new_x)
#     return xs


def plot_single(f, x, max_r=10):
    data = {}
    for i in range(max_r):
        data[i] = []
    for r in range(max_r):
        xs = np.array(near_vector(x, r, 0))
        res = f.evaluate(xs)
        data[r] += res.tolist()
    xs = []
    ys = []
    for i in range(max_r):
        for j in range(len(data[i])):
            xs.append(i)
            ys.append(data[i][j])
    plt.scatter(xs, ys)
    plt.show()


def plot(max_r=20, dim=20, prob_1=0.5):
    t = f'max_r = {max_r}, dim = {dim}, prob_1={prob_1}'
    print(f'{t} start')
    data = {}
    label = {}
    for i in range(max_r):
        data[i] = []
        label[i] = []
    for _ in range(1):
        print(_)
        f = Func(dim, prob_1=prob_1, row=1000, error_rate=0.4)
        x = f.x.copy()
        # x = np.random.randint(0, 2, (dim,), dtype=int)
        print(x, f.x)
        for r in range(max_r):
            xs = np.array(near_vector(x, r, 0))
            res = f.evaluate(xs)
            data[r] += res.tolist()
            d1 = np.abs(xs - f.x)
            hw_distance = np.sum(d1, axis=1)
            # print(hw_distance)
            label[r] += hw_distance.tolist()
    xs = []
    ys = []
    cs = []
    # for i in range(max_r):
    #     xs.append(i)
    #     ys.append(np.mean(data[i]))
    # print(ys)
    # for i in range(max_r):
    #     for j in range(len(data[i])):
    #         xs.append(i)
    #         ys.append(data[i][j])
    #         cs.append(label[i][j])
    # plt.title(t)
    # plt.scatter(xs, ys, marker='o', c=cs, cmap='coolwarm')
    #
    # plt.show()
    # figure, axes = plt.subplots()  # 得到画板、轴
    plt.boxplot(data.values(), patch_artist=True, labels=list(range(max_r)))  # 描点上色
    # set dpi to 200
    plt.xlabel('Hamming distance from the correct secret')
    plt.ylabel('evaluation')
    plt.savefig(f"data/img/{t.replace(' ', '')}.png", dpi=200)
    # plt.show()  # 展示


def plot_with_task(task):
    plot(**task)
    # print(task)


def calculate_near_res(config):
    r = config['r']
    x = config['x']
    a = config['a']
    b = config['b']
    f = config['f']
    xs = np.array(near_vector(x, r, 0))
    res = f.caculate_result(a, b, xs)
    return {'xs': xs, 'res': res}


def t0est_change_x():
    dim = 20
    f = Func(dim, prob_1=0.5, row=int(1e6), error_rate=0.3)
    biga, bigb, bigbf = f.get_some_sample(0.8, 1.1)
    b = np.identity(dim)
    b[-1, :] = 1
    b_reverse = np.abs(np.linalg.inv(b))
    y = np.matmul(b, f.x) % 2
    biga2 = np.matmul(biga, b_reverse) % 2
    random_x = np.random.randint(0, 2, (dim,), dtype=int)
    data = {}
    label = {}
    max_r = dim
    for i in range(max_r):
        data[i] = []
        label[i] = []

    pool = mp.Pool(processes=6)
    configs = []
    for i in range(max_r):
        configs.append({'x': random_x, 'r': i, 'a': biga2, 'b': bigbf, 'f': f})
    # ress = pool.map(calculate_near_res, configs)
    ress = []
    for config in configs:
        ress.append(calculate_near_res(config))

    for r in range(max_r):
        res = ress[r]['res']
        xs = ress[r]['xs']
        data[r] += res.tolist()
        d1 = np.abs(xs - f.x)
        hw_distance = np.sum(d1, axis=1)
        label[r] += hw_distance.tolist()

    xs = []
    ys = []
    cs = []
    for i in range(max_r):
        for j in range(len(data[i])):
            xs.append(i)
            ys.append(data[i][j])
            cs.append(label[i][j])
    plt.scatter(xs, ys, marker='o', c=cs, cmap='coolwarm')

    plt.show()


def main():
    # pool = mp.Pool(processes=6)
    # tasks = []
    # for i in [0.1, 0.2, 0.8, 0.9]:
    #     task = {
    #         'max_r': 3, 'dim': 20, 'prob_1': i
    #     }
    #     tasks.append(task)
    # pool.map(plot_with_task, tasks)
    # pool.close()
    # pool.join()
    # plot(prob_1=0.5, max_r=20, dim=20)
    plot(prob_1=0.15, max_r=20, dim=20)
    # test_change_x()


if __name__ == '__main__':
    main()
