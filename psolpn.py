import psutil

from eva_func import Func
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from logger import logger
import multiprocessing as mp


def plot_single(f, x, a, b, max_r=10):
    logger.debug(f'start plot_single ')
    data = {}
    for i in range(max_r):
        data[i] = []
    for r in range(max_r):
        xs = np.array(near_vector(x, r, 0))
        s = time.time()
        res = f.caculate_result(a, b, xs)
        logger.debug(f'caculate xs.shape:{xs.shape}, use time:{time.time() - s}')
        data[r] += res.tolist()
    # xs = []
    # ys = []
    # for i in range(max_r):
    #     for j in range(len(data[i])):
    #         xs.append(i)
    #         ys.append(data[i][j])
    # plt.scatter(xs, ys)
    # plt.show()
    figure, axes = plt.subplots()  # 得到画板、轴
    axes.boxplot(data.values(), patch_artist=True)  # 描点上色
    # plt.savefig(f"data/img/{t.replace(' ', '')}.png")
    plt.show()  # 展示
    logger.debug('plot finish')


def near_vector(x: np.ndarray, radius: int, start_at: int):
    assert len(x.shape) == 1
    s = time.time()
    xs = []
    if radius < 1:
        return [x]
    if radius > x.shape[0] / 2:
        radius = x.shape[0] - radius
        x = 1 - x
    for d in range(start_at, x.shape[0], 1):
        new_x = x.copy()
        new_x[d] = 1 - new_x[d]
        if radius > 1:
            xs += near_vector(new_x, radius - 1, d + 1)
        else:
            xs.append(new_x)
    if start_at == 0:
        logger.debug(f'near_vector r:{radius}, use time:{time.time() - s}')
    return xs


def hm_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def test_random_hm_distance():
    x = np.random.randint(0, 2, (40,), dtype=int)
    xs = np.random.randint(0, 2, (128, 40), dtype=int)
    d1 = np.abs(xs - x)
    hw_distance = np.sum(d1, axis=1)
    print(hw_distance)


def main(dim=20, error_rate=0.49):
    logger.info(f'start main dim:{dim} error_Rate:{error_rate}')
    mem = psutil.virtual_memory()
    while mem.available < 1 * 1024 * 1024 * 1024:
        time.sleep(1)
    f = Func(dim=dim, error_rate=error_rate, row=int(2 ** 26), prob_1=0.5)
    # print(f.a)
    s = time.time()
    smalla, smallb, smallbf02 = f.get_some_sample(0, 0.25, row_max=100000)
    f.a = None
    # biga, bigb, bigbf = f.get_some_sample(0.95, 1.1)

    # print('end')
    score = f.caculate_result(smalla, smallb, f.x)
    best_score = f.caculate_result(smalla, smallbf02, f.x)
    # print(score)

    # 确认沟壑
    # test_x = np.random.randint(0, 2, (dim,), dtype=int)
    # near1x = np.array(near_vector(test_x, 1, 0))
    # near2x = np.array(near_vector(test_x, 2, 0))
    # near1res = f.caculate_result(biga, bigbf, near1x)
    # near2res = f.caculate_result(biga, bigbf, near2x)
    #
    # near0res = f.caculate_result(biga, bigbf, test_x)
    # near_mean_res = [np.mean(near0res), np.mean(near1res), np.mean(near2res)]
    # print(near_mean_res)
    # if near_mean_res[2] > near_mean_res[1]:
    #     print(f'是指定点')
    #     start_x = test_x.copy()
    # elif near_mean_res[2] < near_mean_res[1]:
    #     print('非指定点')
    #     start_x = near1x[0].copy()
    # else:
    #     # raise BaseException('未知异常')
    #     print('蛇皮点')
    #
    # print(f'HANMING DISTANCE : {hm_distance(f.x, start_x)}')
    # plot_single(f, test_x, biga, bigbf, 20)
    npart = 128
    best_res = -np.inf
    best_x = np.zeros((dim,))
    for step in range(1000):
        xs = np.random.randint(0, 2, (npart, dim), dtype=int)
        ress = f.caculate_result(smalla, smallb, xs)
        for i in range(npart):
            run_flag = True
            while run_flag:
                run_flag = False
                new_xs = np.array(near_vector(xs[i], radius=1, start_at=0))
                new_ress = f.caculate_result(smalla, smallb, new_xs)
                max_index = np.argmax(new_ress)
                if new_ress[max_index] > ress[i]:
                    ress[i] = new_ress[max_index]
                    xs[i] = new_xs[max_index]
                    run_flag = True

        index = np.argmax(ress)
        if ress[index] > best_res:
            best_x = xs[index].copy()
            best_res = ress[index]
        logger.info(f'step:{step} usetime:{time.time() - s} best:{best_res} target:{best_score}')

        if best_score - best_res < 1e-2:
            # real_score = f.evaluate(best_x)
            logger.info(f'find:{np.sum(np.abs(best_x - f.x)) == 0} findx:{best_x} real_x:{f.x}')
            if np.sum(np.abs(best_x - f.x)) == 0:
                logger.info('find')
                return True
            break
    return False


def mutation(x):
    dim = x.shape[0]
    indexs = random.sample(range(dim), 2)
    for index in indexs:
        x[index] = 1 - x[index]
    return x


if __name__ == '__main__':
    logger.info('start run')
    data = {}
    pool = mp.Pool(processes=2)
    for dim in range(10, 12, 1):
        success = 0
        all_run = 3
        ress = pool.map(main, [dim] * all_run)
        for i in range(all_run):
            if ress[i]:
                success += 1
        data[dim] = success / all_run
        if data[dim] == 0:
            if data.get(dim - 1, 1) == 0:
                break
    print(data)
    # test_random_hm_distance()
