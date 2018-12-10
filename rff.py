
import sys
import numpy as np
import pinfo


class RandomFourierFeature:

    # w is a d-dimensional vector
    def pdf(self, w):
        if self.kernel == "G":
            # (2pi)^(-D/2) e^(-0.5 ||w||_2^2)
            return ((2 * np.pi) ** (-1 * self.D / 2)) * np.exp(-0.5 * (np.linalg.norm(w) ** 2))
        elif self.kernel == "L":
            # (1 / pi^d) / [(1 + w_1^2) * (1 + w_2^2) * ... ]
            vals = np.asarray([ 1 + (component ** 2) for component in w ])
            return np.prod(vals) / (np.pi ** self.d)
        elif self.kernel == "C":
            return np.exp(-1 * np.abs(w))

    # Monte-Carlo Rejection Sampling (https://en.wikipedia.org/wiki/Rejection_sampling)
    def sample(self, interval):
        # computes max value of PDF; specific to kernel
        pdf_max = 1
        if self.kernel == "G":
            pdf_max = ((2 * np.pi) ** (-1 * self.D / 2))
        elif self.kernel == "L":
            pdf_max = 1 / (np.pi ** self.d)
        elif self.kernel == "C":
            pdf_max = 1

        while True:
            w = np.random.rand(1) * (interval[1] - interval[0]) + interval[0]
            y = np.random.rand(1) * pdf_max

            if y <= self.pdf(w):
                return w

    def d_sample(self, d):
        return [self.sample((-10, 10)) for _ in range(self.d)]

    def __init__(self, *args):
        if len(args) == 0 and type(args[0]) == str:
            self.__init__load(args[0])
        elif len(args) == 3:
            self.__init__new(*args)
        else:
            raise Exception("Invalid RFF Initialization.")

    def __init__load(self, file):
        timer = pinfo.Task()
        self.d, self.D, self.b, self.W = np.load(file)
        timer.stop(
            "{desc} loaded from {f}"
            .format(desc=self.__str__(), f=file))

    def __init__new(self, d, D, kernel):

        timer = pinfo.Task()
        self.d = d
        self.D = D
        self.kernel = kernel
        self.b = np.random.uniform(0, 2 * np.pi, D)
        if self.kernel == "G":
            random_vectors = [np.random.normal(0, 1, d) for _ in range(D)]
            self.W = np.array([vector * (self.sample((-10, 10)) / np.linalg.norm(vector)) for vector in random_vectors], dtype=np.float32)
        elif self.kernel == "L" or self.kernel == "C":
            self.W = np.reshape(np.array([self.d_sample(self.d) for _ in range(self.D)]), (D, d))
        timer.stop(
            "{desc} created"
            .format(desc=self.__str__()), self.W, self.b)

    def transform(self, x):
        return np.sqrt(2 / self.D) * np.cos(np.dot(self.W, x) + self.b)

    def __size(self):
        return "{s:.2f}MB".format(
            s=(sys.getsizeof(self.b) + sys.getsizeof(self.W)) / 10**6)

    def __str__(self):
        return (
            "{d}->{D} Random Fourier Feature"
            .format(d=self.d, D=self.D))

    def save(self, file='rff'):
        timer = pinfo.Task()
        np.save(file, [self.d, self.D, self.b, self.W])
        timer.stop(
            "{desc} saved to {f}"
            .format(desc=self.__str__()))
