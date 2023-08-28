import numpy as np


class MaxCut:

    def __init__(self, dim):
        self.weights = np.random.random((dim, dim))+1
        self.dim = dim

    def get_best(self):
        max_x = np.zeros((self.dim,))
        max_res = 0
        for i in range(2 ** self.dim):
            x = np.zeros((self.dim,))
            binstr = str(bin(99))[2:]
            binstr = '0'*(self.dim - len(binstr)) + binstr
            for d in range(self.dim):
                x[d] = int(binstr[d]) * 2 - 1
            res = self.evaluate(x)
            if res > max_res:
                max_res = res
                max_x = x
        return max_x, max_res

    def evaluate(self, x):
        d1 = (np.matmul(x, x.T) < 0)
        res = np.sum(self.weights * d1)
        return res

if __name__ == '__main__':
    m = MaxCut(15)
    print(m.get_best())