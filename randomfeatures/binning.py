
"""Random Binning Feature"""

import numpy as np
import math
from syllabus import Task
from .sample import sample, ft_laplacian


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

    def __init__(self, d, D, cores=None, task=None):

        self.d = d
        self.D = D

        if task is None:
            task = Task()
        task.reset()
        task.set_info(name='Random Binning Feature', desc=self.__str__())

        def get_p_set(args):
            """Generate one set of deltas (bin width) and mus (bin offset)

            Parameters
            ----------
            args : (int, Task)
                [0] Number of dimensions
                [1] Task parent

            Returns
            -------
            (np.array, np.array)
                [0] delta vector for this feature
                [1] mu vecotr for this feature
            """
            n, task = args

            delta_p = sample(ft_laplacian, n)
            mu_p = [np.random.uniform(0, delta_m) for delta_m in delta_p]

            task.done('', slient=True)
            return (delta_p, mu_p)

        gen = task.pool(self.get_p_set, [d for _ in range(D)])

        self.delta = np.array([x[0] for x in gen], dtype=np.float32)
        self.mu = np.array([x[1] for x in gen], dtype=np.float32)

        task.done(
            "{desc} created"
            .format(desc=self.__str__()), self.delta, self.mu)

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
