

import numpy as np
import random
from sklearn.cluster import KMeans
from syllabus import Task


class CKM:
    """Color K Means Transform

    Parameters
    ----------
    n : int
        Number of means
    dataset : IDCDataset object
        Dataset to run on
    task : Task
        Task to register under
    """

    def __init__(self, n, dataset, task=None):

        if task is None:
            task = Task()
        task.start(name='CKM', desc="Color K Means Clustering")

        self.n = n

        knndata = []
        for d in dataset.data:
            d = d.reshape(50, 50, 3)
            for i in range(10):
                x = random.randint(0, 49)
                y = random.randint(0, 49)
                knndata.append(d[x, y])

        self.kmeans = KMeans(n_clusters=n).fit(knndata).cluster_centers_

        task.done(desc="Generated K Means model (n={n})".format(n=self.n))

    def cluster(self, color):
        """Find which cluster a color belongs to

        Parameters
        ----------
        color : np.array (shape=(3))
            Single pixel to classify

        Returns
        -------
        int
            Index of the closest cluster
        """
        d = [np.linalg.norm(c - color, ord=2) for c in self.kmeans]
        return np.argmin(d)

    def map(self, img):
        """Map an image to a color histogram

        Parameters
        ----------
        img : np.array (shape=(N, N, 3))
            Array to compute histogram for

        Returns
        int[]
            Array of frequencies of each color
        """
        fvec = [0 for _ in range(len(self.kmeans))]
        for p in img:
            fvec[self.cluster(p)] += 1
        return fvec
