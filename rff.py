
import sys
import numpy as np
import pinfo


class RandomFourierFeature:

    def __init__(self, *args):
        if len(args) == 0 and type(args[0]) == str:
            self.__init__load(args[0])
        elif len(args) == 2:
            self.__init__new(*args)
        else:
            raise Exception("Invalid RFF Initialization.")

    def __init__load(self, file):
        timer = pinfo.Task()
        self.d, self.D, self.b, self.W = np.load(file)
        timer.stop(
            "{desc} loaded from {f}"
            .format(desc=self.__str__(), f=file))

    def __init__new(self, d, D):

        timer = pinfo.Task()
        self.d = d
        self.D = D
        self.b = np.random.uniform(0, 2 * np.pi, D)
        self.W = np.array(
            [np.random.normal(0, 1, d) for _ in range(D)], dtype=np.float32)
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
