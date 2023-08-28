import json
import os
import multiprocessing as mp

from common.get_good_model import get_good_model
from eva_func import Func
from optimizer.EDAPSOGA.eda_pso_ga import EDAPSOGA
from optimizer.GA.ga import GA
from optimizer.NBOA.rlnboa import RLNBOA
import numpy as np
import glob
from pathlib import Path

from optimizer.BGWOPSO.bgwopso import bgwopso
from optimizer.BPSO.bpso import bpso
from optimizer.disABC.disABC import disABC
from optimizer.random_search.rs import RS
import hashlib
import copy

# normal_optis = [bgwopso, bpso, disABC, RS, EDAPSOGA, GA]
normal_optis = [bgwopso, bpso, RS, EDAPSOGA, GA]

TASK_RES_DIR = Path(r'evaluate/task_result')


def md5(s):
    m = hashlib.md5()
    b = s.encode(encoding='utf-8')
    m.update(b)
    str_md5 = m.hexdigest()
    return str_md5


def evluate_optimizer(config):
    print(f'evluate_optimizer config:{config}')
    task_ID = config.get('task')
    task_md5 = config.get('task_md5')
    dim = config['dim']
    cls = config['class']
    model = config.get('model')
    npart = config['npart']
    prob_1 = config.get('prob_1', 0.5)
    fes = 2 ** dim // 4
    file_name = TASK_RES_DIR.joinpath('{}/task{}.json'.format(task_md5, task_ID))
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r', encoding='UTF-8') as f:
                json_str = f.read()
                res = json.loads(json_str)
                return res
        except Exception as e:
            print('get cache error:', e, file_name)
    func = Func(dim, prob_1=prob_1)
    test_fun = func.evaluate
    best_value = func.best_value[0]
    hyperparams = {
        'best_value': best_value
    }
    if model:
        optimizer = cls(fes, test_fun, npart, dim, hyperparams=hyperparams, model_file=model)
    else:
        optimizer = cls(fes, test_fun, npart, dim, hyperparams=hyperparams)
    optimizer.run()
    res = {
        'record': optimizer.record,
        'best': best_value,
        'gbest': optimizer.get_best_value(),
        'task': task_ID,
        'class': optimizer.name,
        'dim': dim
    }
    print(
        f'{optimizer.name}|gbest:{optimizer.get_best_value()}|best:{best_value}|fes:{optimizer.max_fes}|npart:{optimizer.npart}|ndim:{optimizer.ndim}')

    json_str = json.dumps(res)
    with open(file_name, 'w', encoding='UTF-8') as f:
        f.write(json_str)
    return res


def evluate_rlnboa(config):
    dim = config['dim']
    model = config['model']
    npart = config['npart']
    run_times = config.get('run_times', 10)
    ress = []
    for _ in range(run_times):
        fes = 2 ** dim // 10
        func = Func(dim)
        test_fun = func.evaluate
        best_value = np.int(func.best_value[0])
        nboa = RLNBOA(fes, test_fun, npart, dim, model_file=model)
        nboa.run()
        res = {
            'record': nboa.record,
            'best': best_value,
            'gbest': nboa.gbest_fit,
        }
        ress.append(res)
    json_str = json.dumps(ress)
    file_name = './data/RLNBOA-DIM{}.json'.format(dim)
    if os.path.exists(file_name):
        print('实验已经做过 跳过')
        return
    with open(file_name, 'w', encoding='UTF-8') as f:
        f.write(json_str)
    # return ress


def evaluate_dim(processes=4):
    pool = mp.Pool(processes=processes)
    test_configs = []
    for dim in range(5, 20, 1):
        config = {
            'dim': dim,
            'model': R'D:\develop\probcaculate\rl\train0\ddpg_actor_episode100.h5',
            'npart': dim // 5 * 20,
            'run_times': 10
        }
        test_configs.append(config)
    test_configs.reverse()
    pool.map(evluate_rlnboa, test_configs)


def evaluate_model(dims, processes=4, models=None, runtimes=1, prob_1=0.5):
    if models is None:
        models = glob.glob(r"../rl/*/ddpg_actor_episode**.h5")
    tasks = []

    for dim in dims:
        for model in models:
            task = {
                'dim': dim,
                'class': RLNBOA,
                'model': model,
                'npart': 100,
                'prob_1': prob_1,
            }
            for _ in range(runtimes):
                tasks.append(task.copy())

    for i in range(len(tasks)):
        tasks[i]['task'] = i

    json_tasks = copy.deepcopy(tasks)
    for i in range(len(json_tasks)):
        json_tasks[i]['class'] = json_tasks[i]['class'].__name__
    task_str = json.dumps(json_tasks)
    task_md5 = md5(task_str)

    # 设置task目录
    dir_path = '{}/{}/'.format(TASK_RES_DIR, task_md5)
    print(dir_path)
    if os.path.exists(dir_path):
        pass
    else:
        os.mkdir(dir_path)
    with open('{}/task.json'.format(dir_path), 'w') as f:
        f.write(task_str)

    for i in range(len(tasks)):
        tasks[i]['task_md5'] = task_md5

    if processes:
        print('多进程')
        pool = mp.Pool(processes=processes)
        ress = pool.map(evluate_optimizer, tasks)
    else:
        print('单进程')
        ress = []
        for task in tasks:
            ress.append(evluate_optimizer(task))

    data = {}

    for res in ress:
        dim = res['dim']
        name = res['class']
        set_dict(data, [name, dim], res)

    with open('test_result.txt', 'a') as f:
        print(f'task ned prob1:{prob_1} dim:{dim}', file=f)
    data_process(data)
    json_str = json.dumps(data)
    with open(r'data/json.json', 'w') as f:
        f.write(json_str)


def evaluate_all(processes=4):
    tasks = []
    task_dic = {}
    runtimes = 10
    start_dim, enddim = 12, 25
    # 其他算法

    for opti in normal_optis:
        for dim in range(start_dim, enddim, 1):
            task = {
                'dim': dim,
                'class': opti,
                'npart': 100,
            }
            for _ in range(runtimes):
                tasks.append(task)

    # RLNBOA
    for dim in range(start_dim, enddim, 1):
        task = {
            'dim': dim,
            'class': RLNBOA,
            'model': r'../rl/train20/ddpg_actor_episode100.h5',
            'npart': 100,
        }
        for _ in range(runtimes):
            tasks.append(task)

    for i in range(len(tasks)):
        tasks[i]['task'] = i
        task_dic[i] = tasks[i]

    if processes:
        pool = mp.Pool(processes=processes)
        ress = pool.map(evluate_optimizer, tasks)
    else:
        ress = []
        for task in tasks:
            ress.append(evluate_optimizer(task))

    data = {}

    for res in ress:
        dim = res['dim']
        name = res['class']
        set_dict(data, [name, dim], res)

    data_process(data)
    json_str = json.dumps(data)
    with open(r'data/json.json', 'w') as f:
        f.write(json_str)


def data_process(data):
    with open('test_result.txt', 'a') as f:
        for name, d1 in data.items():
            for dim, ress in d1.items():
                success_rate = 0
                record_len = len(ress)
                for i in range(record_len):
                    record = ress[i]
                    best = record['gbest']
                    target = record['best']
                    success_rate += 1 * (best == target)
                success_rate /= record_len
                print(f'name:{name}\t dim:{dim}\t success:{success_rate}', file=f)


def set_dict(dic, keys, value):
    for key in keys[:-1]:
        if key in dic:
            dic = dic[key]
        else:
            dic[key] = {}
            dic = dic[key]
    key = keys[-1]
    if key in dic:
        pass
    else:
        dic[key] = []
    dic[key].append(value)


print('evaluate')
if __name__ == '__main__':
    print('start')
    dims = [30]

    evaluate_model(dims=dims, processes=24, models=get_good_model('../data/good_model_t100.txt'), runtimes=20,
                   prob_1=0.5)
    evaluate_model(dims=dims, processes=24, models=get_good_model('../data/good_model_t100.txt'), runtimes=20,
                   prob_1=0.4)
    evaluate_model(dims=dims, processes=24, models=get_good_model('../data/good_model_t100.txt'), runtimes=20,
                   prob_1=0.3)
    print('end')

    # task = {
    #     'task': 0,
    #     'dim': 15,
    #     'class': GA,
    #     'npart': 100,
    # }
    # res = evluate_optimizer(task)
    # print(res['gbest'], res['best'])
