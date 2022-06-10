import numpy as np
from scipy.optimize import minimize


def f(x, y):
    return 2.107 * x + 2.3 * y


res = minimize(
    lambda args: -f(*args),
    np.zeros((1, 2)),
    constraints={'type': 'eq', 'fun': lambda args: args[0] + args[1] - 1e4},
    bounds=[(0, 1e4), (0, 1e4)])


if __name__ == '__main__':
    print(res.x)
    print(-res.fun)
