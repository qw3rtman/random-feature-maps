
import numpy as np
import random
from sklearn.cluster import KMeans
import idc
import pinfo


class CKM:

    def __init__(self, n, dataset):

        timer = pinfo.Task()

        self.n = n

        knndata = []
        for d in dataset.data:
            d = d.reshape(50, 50, 3)
            for i in range(10):
                x = random.randint(0, 49)
                y = random.randint(0, 49)
                knndata.append(d[x, y])

        self.kmeans = KMeans(n_clusters=n).fit(knndata).cluster_centers_

        timer.stop("Generated K Means model (n={n})".format(n=self.n))

    def cluster(self, color):
        d = [np.linalg.norm(c - color, ord=2) for c in self.kmeans]
        return np.argmin(d)

    def map(self, img):
        fvec = [0 for _ in range(len(self.kmeans))]
        for p in img:
            fvec[self.cluster(p)] += 1
        return fvec


def get_idc_colors(n):
    return CKM(n, idc.IDCDataset(idc.PATIENTS, p=0.01))
