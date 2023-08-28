import numpy as np


def test_fit(xs, func_num=0):
    xs = xs * 1
    if len(xs.shape) == 1:
        xs = xs.reshape(1, -1)
    elif len(xs.shape) == 2:
        pass

    result = None

    if func_num == 0:
        # ONE MAX
        result = np.sum(xs, axis=1)

    elif func_num == 1:
        # NOISY ONE MAX
        result = np.sum(xs, axis=1)
        result = result + np.random.normal(loc=0, scale=31, size=result.shape)

    elif func_num == 2:
        # Bounded deceptive
        xs = xs.reshape(xs.shape[0], -1, 4)
        s = np.sum(xs, axis=2)
        s = (s < 4) * (3 - s) + (s == 4) * 4
        s = np.sum(s, axis=1)
        result = s

    if result is None:
        print(f'error func_num:{func_num},xs:{xs}')
        exit()

    return result


if __name__ == '__main__':
    a = np.ones(100, dtype=np.bool)
    print(test_fit(a, 1))
