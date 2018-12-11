
"""Random Fourier Feature"""

import numpy as np
from syllabus import Task

from .sample import KERNELS, sample, sample_1d


class RandomFourierFeature:
    """Random Fourier Feature

    Parameters
    ----------
    d : int
        Input space dimension
    D : int
        Feature space dimension
    kernel : char
        Kernel to use; 'G', 'L', or 'C'

    References
    ----------
    ..  [1] A. Rahimi, B. Recht, "Random Features for Large-Scale Kernel
        Machines"
    """

    def __init__(self, d, D, kernel='G', task=None):

        self.d = d
        self.D = D

        kernel = kernel.upper()
        if kernel not in ['G', 'L', 'C']:
            raise Exception('Invalid Kernel')
        self.kernel = kernel

        if task is not None:
            task = Task()
        task.reset()
        task.set_info(name='Random Fourier Feature', desc=self.__str__())

        # Create feature
        self.create()

        task.done(
            "{desc} created"
            .format(desc=self.__str__()), self.W, self.b)

    def create(self):
        """Create a d->D fourier random feature"""

        self.b = np.random.uniform(0, 2 * np.pi, self.D)
        if self.kernel == 'G':
            random_vectors = [
                np.random.normal(0, 1, self.d) for _ in range(self.D)]
            self.W = np.array([
                vector *
                (
                    sample_1d(KERNELS[self.kernel], [-10, 10]) /
                    np.linalg.norm(vector)
                )
                for vector in random_vectors
            ], dtype=np.float32)
        else:
            self.W = np.reshape(
                np.array([
                    sample(KERNELS[self.kernel], self.d)
                    for _ in range(self.D)]),
                (self.D, self.d))

    def transform(self, x):
        """Transform a vector using this feature

        Parameters
        ----------
        x : np.array (shape=(d))
            Array to transform; must be single dimension vector

        Returns
        -------
        x : np.array (shape=(D))
            Feature space transformation of x
        """
        return np.sqrt(2 / self.D) * np.cos(np.dot(self.W, x) + self.b)

    def __str__(self):
        """Get string representation

        Shown as "<d>-><D> Random Fourier Feature"
        """
        return (
            "{d}->{D} Random Fourier Feature"
            .format(d=self.d, D=self.D))
