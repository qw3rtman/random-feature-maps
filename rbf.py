

import math
import numpy as np
import pinfo

from multiprocessing import Pool


def sample(pdf, pdfmax):

    while True:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, pdfmax)
        if pdf(x) < y:
            return x


def laplacian(x):
    return x * math.exp(-x)


class RandomBinningFeature:

    def __init__(self, n, P, cores=None):

        self.n = n
        self.P = P

        timer = pinfo.Task("Creating Random Binning Feature...")

        p = Pool(cores)
        gen = p.map(self.get_p_set, [n for _ in range(P)])

        self.delta = np.array([x[0] for x in gen], dtype=np.float32)
        self.mu = np.array([x[1] for x in gen], dtype=np.float32)

        timer.stop(
            "{desc} created"
            .format(desc=self.__str__()), self.delta, self.mu)

    def get_p_set(self, n):
        delta_p = [sample(laplacian, 1 / math.e) for _ in range(n)]
        mu_p = [np.random.uniform(0, delta_m) for delta_m in delta_p]
        return (delta_p, mu_p)

    def transform(self, x):

        ret = []
        for mu_p, delta_p in zip(self.mu, self.delta):
            tmp = [0 for i in range(128)]
            for x_i, mu, delta in zip(x, mu_p, delta_p):
                tmp[math.ceil((x_i - mu) / delta) % 128] += 1
            ret += tmp

        return np.array(ret, dtype=np.uint8)

        #return np.array([
        #    (sum([
        #        1 >> math.ceil((x_i - mu) / delta)
        #        for x_i, mu, delta in zip(x, mu_p, delta_p)
        #    ]) % 2**32)
        #    for mu_p, delta_p in zip(self.mu, self.delta)],
        #    dtype=np.uint32)

    def __str__(self):
        return (
            "{d}->{D} Random Binning Feature"
            .format(d=self.n, D=self.P))
