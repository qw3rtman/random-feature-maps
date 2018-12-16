
"""Random Binning Feature"""

import numpy as np
import math
from syllabus import Task
from .sample import sample, ft_laplacian
from .raw_array import make_raw


def get_p_set(args):
    """Generate one set of deltas (bin width) and mus (bin offset)

    Parameters
    ----------
    n : int
        Number of dimensions
    task : Task
        task parent

    Returns
    -------
    (np.array, np.array)
        [0] delta vector for this feature
        [1] mu vector for this feature
    """

    n, task = args

    delta_p = sample(ft_laplacian, n)
    mu_p = [np.random.uniform(0, delta_m) for delta_m in delta_p]

    if task is not None:
        task.done(silent=True)
    return (delta_p, mu_p)


class RandomBinningFeature:
    """Random Binning Feature

    Parameters
    ----------
    d : int
        Input space dimension
    D : int
        Feature space dimension; actual dimension is 128 * D binary array
    cores : int
        Number of cores to use for generation
    task : Task or none
        Task to register feature generation under
    """

    def __init__(self, d, D, delta=None, mu=None, cores=None, task=None):

        self.d = d
        self.D = D

        if delta is None or mu is None:
            self.__new(task, cores)
        else:
            self.__load(delta, mu)

    def __load(self, delta, mu):
        self.delta = np.frombuffer(
            delta, dtype=np.float32).reshape([self.D, self.d])
        self.mu = np.frombuffer(
            mu, dtype=np.float32).reshape([self.D, self.d])

    def __new(self, task, cores):

        if task is None:
            task = Task()
        task.start(name='Random Binning Feature', desc=self.__str__())

        gen = task.pool(
            get_p_set, [self.d for _ in range(self.D)],
            cores=cores, process=True, name='Random Binning Feature')

        self.delta = np.array([x[0] for x in gen], dtype=np.float32)
        self.mu = np.array([x[1] for x in gen], dtype=np.float32)

        task.done(
            self.delta, self.mu,
            desc="{desc} created".format(desc=self.__str__()))

    def mp_package(self):

        if not hasattr(self, 'delta_raw') or not hasattr(self, 'mu_raw'):
            self.delta_raw = make_raw(self.delta)
            self.mu_raw = make_raw(self.delta)

        return (self.d, self.D, self.delta_raw, self.mu_raw)

    def transform(self, x):
        """Transform a vector using this feature

        Parameters
        ----------
        x : np.array (shape=(d))
            Array to transform; must be a single dimension vector

        Returns
        -------
        x : np.array (shape=(D))
            Feature space transformation of x
        """
        ret = []
        for mu_p, delta_p in zip(self.mu, self.delta):
            tmp = [0 for i in range(128)]
            for x_i, mu, delta in zip(x, mu_p, delta_p):
                tmp[math.ceil((x_i - mu) / delta) % 128] += 1
            ret += tmp

        return 1 / np.sqrt(self.D) * np.array(ret, dtype=np.uint8)

    def __str__(self):
        """Get String representation

        Shown as "<d>-><D> Random Binning Feature"
        """
        return (
            "{d}->{D} Random Binning Feature"
            .format(d=self.d, D=self.D))
