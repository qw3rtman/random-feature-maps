

import math
import numpy as np
import pinfo


def sample(pdf, pdfmax):

    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, pdfmax)
        if pdf(x) < y:
            return x


def laplacian(x):
    return x * math.exp(-x)


class RandomBinningFeature:

    def __init__(self, n, P):

        self.n = n
        self.P = P

        timer = pinfo.Task("Creating Random Binning Feature...")
        self.delta = np.array([
            [sample(laplacian, 1 / math.e) for j in range(n)]
            for i in range(P)
        ], dtype=np.float32)

        self.mu = np.array([
            [np.random.uniform(0, delta_m) for delta_m in delta_p]
            for delta_p in self.delta], dtype=np.float32)
        timer.stop(
            "{desc} created"
            .format(desc=self.__str__()), self.delta, self.mu)

    def transform(self, x):

        return np.array([
            (sum([
                math.ceil((x_i - mu) / delta)
                for x_i, mu, delta in zip(x, mu_p, delta_p)
            ]) % 2**16) / 2**16
            for mu_p, delta_p in zip(self.mu, self.delta)],
            dtype=np.float32)

    def __str__(self):
        return (
            "{d}->{D} Random Binning Feature"
            .format(d=self.n, D=self.P))
