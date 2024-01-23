import numpy as np
import time

from eva_func import Func


def solve(f: Func, sample_num=10):
    start_time = time.time()
    dim = f.dim
    row = f.row
    for _ in range(sample_num):
        a, b = get_matrix(dim, f, row, full_rank=False)
        x = np.zeros((dim,), dtype=int)
        for i in range(dim - 1, -1, -1):
            if a[i][i] == 0:
                continue
            x[i] = b[i]
            for j in range(i + 1, dim):
                x[i] = (x[i] + a[i][j] * x[j]) % 2
    use_time = time.time() - start_time
    use_time = use_time / sample_num
    all_right_prob = (1 - f.error_rate) ** dim
    full_rank_prob = 1
    prob = all_right_prob * full_rank_prob
    real_use_time = use_time / prob
    return real_use_time


def get_matrix(dim, f, row, full_rank=True):
    run_time = 0
    run_flag = True
    while run_flag:
        run_time += 1
        assert run_time < 10000
        indexs = np.random.randint(0, row, (dim,))
        a = f.a[indexs]
        ori_a = a.copy()
        b = f.b[indexs]
        ori_b = b.copy()
        # solve Ax=b with gauss
        for i in range(dim):
            if a[i][i] == 0:
                for j in range(i + 1, dim):
                    if a[j][i] == 1:
                        a[i], a[j] = a[j], a[i]
                        b[i], b[j] = b[j], b[i]
                        break
            for j in range(i + 1, dim):
                if a[j][i] == 1:
                    a[j] = (a[j] + a[i]) % 2
                    b[j] = (b[j] + b[i]) % 2
        if full_rank:
            for i in range(dim):
                if a[i][i] == 0:
                    run_flag = True
                    break
                else:
                    run_flag = False
        else:
            run_flag = False
    return a, b

def number_of_query_to_invertible(dim=1):
    noq = 0
    # random generate 0 1 matrix
    matrix = np.random.randint(0, 2, (dim, dim))
    noq = noq + dim
    while True:
        # check if matrix is invertible
        for i in range(dim):
            if matrix[i][i] == 0:
                for j in range(i + 1, dim):
                    if matrix[j][i] == 1:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        break
            for j in range(i + 1, dim):
                if matrix[j][i] == 1:
                    matrix[j] = (matrix[j] + matrix[i]) % 2
        invertible = True
        for i in range(dim):
            if matrix[i][i] == 0:
                noq += 1
                # regenerate matrix[i]
                matrix[i] = np.random.randint(0, 2, (dim,))
                invertible = False
        if invertible:
            print(f'noq:{noq} dim:{dim}')
            break




def random_matrix_invertible(dim=1, test_num=1):
    invertible_num = 0
    for i in range(test_num):
        # random generate 0 1 matrix
        matrix = np.random.randint(0, 2, (dim, dim))
        # check if matrix is invertible
        for i in range(dim):
            if matrix[i][i] == 0:
                for j in range(i + 1, dim):
                    if matrix[j][i] == 1:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        break
            for j in range(i + 1, dim):
                if matrix[j][i] == 1:
                    matrix[j] = (matrix[j] + matrix[i]) % 2
        invertible = True
        for i in range(dim):
            if matrix[i][i] == 0:
                invertible = False
                break
        if invertible:
            invertible_num += 1
    print(f'dim:{dim} invertible_num:{invertible_num} test_num:{test_num} prob:{invertible_num / test_num:.4f} prob2:{invertible_num / test_num * 2 ** dim:.4f}')


if __name__ == '__main__':
    time_dict = {}
    for dim in range(50):
        f = Func(dim=dim, error_rate=0.49, row=int(2 ** 10), prob_1=0.5)
        time_dict[dim] = solve(f)
    print(time_dict)
    # save time dict to csv
    with open('time_dict.csv', 'w') as f:
        for k, v in time_dict.items():
            f.write(f'{k},{v}\n')

    # for i in range(10, 50):
    #     number_of_query_to_invertible(dim=i)
