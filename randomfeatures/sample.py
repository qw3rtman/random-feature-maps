

import numpy as np


def ft_gaussian(x):
    """FT of gaussian, normalized so that max_x pdf(x) = 1"""
    return (np.exp(-0.5 * (np.linalg.norm(x) ** 2)))


def ft_laplacian(x):
    """FT of laplacian, normalized so that max_x pdf(x) = 1"""
    return np.prod(np.asarray([1 + x_i ** 2] for x_i in x))


def ft_cauchy(x):
    """FT of cauchy, normalized so that max_x pdf(x) = 1"""
    return np.exp(-1 * np.abs(x))


KERNELS = {
    'G': ft_gaussian,
    'L': ft_laplacian,
    'C': ft_cauchy
}


def sample_1d(pdf, interval):
    """Monte-Carlo rejection sampling"""
    while True:
        w = np.random.rand(1) * (interval[1] - interval[0]) + interval[0]
        y = np.random.rand(1)

        if y <= pdf(w):
            return w


def sample(pdf, d):
    """Monte Carlo Rejection Sampling"""
    return [sample_1d(pdf, [-10, 10]) for _ in range(d)]
