
"""Random Fourier Feature"""

import numpy as np
from syllabus import Task

from .raw_array import make_raw
from .sample import KERNELS, sample, sample_1d


class RandomFourierFeature:
    """Random Fourier Feature

    Parameters
    ----------
    d : int
        Input space dimension
    D : int
        Feature space dimension
    W : RawArray or None
        If not None, used as the transformation matrix W instead of
        generating a new matrix
    b : RawArray or None
        If not None, used as the offset coefficients b
    kernel : char
        Kernel to use; 'G', 'L', or 'C'

    References
    ----------
    ..  [1] A. Rahimi, B. Recht, "Random Features for Large-Scale Kernel
        Machines"
    """

    def __init__(self, d, D, W=None, b=None, kernel='G', task=None):

        self.d = d
        self.D = D

        kernel = kernel.upper()
        if kernel not in ['G', 'L', 'C']:
            raise Exception('Invalid Kernel')
        self.kernel = kernel

        if W is None or b is None:
            self.__new(task)
        else:
            self.__load(W, b)

    def __load(self, W, b):
        """Load from existing RawArrays"""

        self.W = np.frombuffer(W, dtype=np.float32).reshape([self.D, self.d])
        self.b = np.frombuffer(b, dtype=np.float32)

    def __new(self, task):
        """Create new W and b"""

        if task is None:
            task = Task()
        task.start(name='Random Fourier Feature', desc=self.__str__())

        # Create feature
        self.create()

        task.done(
            self.W, self.b, desc="{desc} created".format(desc=self.__str__()))

    def mp_package(self):
        """Package into a multiprocessing-ready RawArray

        Returns
        -------
        (int, int, np.array, np.array)
            [0] input dimension (d)
            [1] output dimension (D)
            [2] W matrix (shape=(D, d))
            [3] b matrix (shape=(D))
        """

        if not hasattr(self, 'W_raw') or not hasattr(self, 'b_raw'):
            self.W_raw = make_raw(self.W)
            self.b_raw = make_raw(self.b)

        return (self.d, self.D, self.W_raw, self.b_raw)

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
                    for _ in range(self.D)], dtype=np.float32),
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
