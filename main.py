import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 0


def integral(f, x):
    y = f(x) * (x[1] - x[0])
    return y.dot(np.ones((y.shape[0])))


def series(f, domain: np.array, order: int):
    assert domain.ndim == 1, 'the dimention of the domain has to be 1, got {}'.format(domain.ndim)
    domain_length = domain[-1] - domain[0]
    _l = domain_length / 2

    def even_factor_func(x, order):
        return f(x) * np.cos(order * np.pi * x / _l) / _l

    def odd_factor_func(x, order):
        return f(x) * np.sin(order * np.pi * x / _l) / _l

    _a = []
    _b = []
    _k = np.linspace(1, order, order, dtype=np.float32)
    for k in _k:
        # print("k = {}".format(k))
        def int_fun_odd(x):
            return odd_factor_func(x, k)

        def int_fun_even(x):
            return even_factor_func(x, k)

        _a.append(
            integral(int_fun_even, domain))
        _b.append(
            integral(int_fun_odd, domain))

    a_0 = integral(f, domain) / _l
    _, k = np.meshgrid(domain, _k)
    _, a = np.meshgrid(domain, _a)
    x, b = np.meshgrid(domain, _b)
    Y = a * np.cos(np.pi * k * x / _l) + \
        b * np.sin(np.pi * k * x / _l)
    # return Y,a_0
    y = np.ones(Y.shape[0]).dot(Y)
    return y + a_0/2


if __name__ == '__main__':
    def step(x):
        return ((np.sign(x) + 1)/2)

    def tri(x):
        return (1-np.abs(x))

    def func(x):
        return step(x + 0.5)

    f = func
    x = np.linspace(-1, 1, 500)
    plt.plot(x, f(x))
    for k in range(5):
        y = series(f, x, 2*k + 1)
        plt.plot(x, y, label='k={}'.format(2*k+1))
    plt.legend()
    plt.show()
    pass
