import numpy as np
import math


def CmbinationNumber(n, m):
    a = math.factorial(n) / (math.factorial(n - m) * math.factorial(m))
    return a


def main():
    for dim in range(20, 41, 1):
        prob = 0
        for i in range(0, dim // 5, 1):
            prob += CmbinationNumber(dim, i) / 2 ** dim
        print(f'dim:{dim} prob:{prob * 1e6} need_sample:{1000 / prob} {np.log2(1000/prob)}')




if __name__ == '__main__':
    main()
