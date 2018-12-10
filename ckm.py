
import numpy as np

COLORS = [
    [0.80644852, 0.38029161, 0.53791945],
    [0.91839878, 0.85894818, 0.89004843],
    [0.87836307, 0.62515271, 0.72896611],
    [0.36270132, 0.20171265, 0.39420037],
    [0.68054873, 0.52128964, 0.66823531],
    [0.86655914, 0.5533991, 0.67566496],
    [0.54068249, 0.35905193, 0.55175357],
    [0.94786119, 0.93404472, 0.94099881],
    [0.77535056, 0.51966659, 0.66664663],
    [0.89454895, 0.70014632, 0.78256955],
    [0.81042716, 0.67927693, 0.77468111],
    [0.1887369, 0.09174647, 0.1499772],
    [0.70946824, 0.26137179, 0.36685741],
    [0.76837261, 0.59705625, 0.72106675],
    [0.66111113, 0.3456741, 0.52462614],
    [0.61506915, 0.43247203, 0.60552737],
    [0.89510474, 0.77455955, 0.83494979],
    [0.84309752, 0.47121092, 0.61452705],
    [0.46873258, 0.28578547, 0.48623796],
    [0.72541516, 0.43962849, 0.60756796],
]


def cluster(color):
    d = [np.linalg.norm(c - color, ord=2) for c in COLORS]
    return np.argmin(d)


def map(image):
    fvec = [0 for _ in range(len(COLORS))]
    for p in image:
        fvec[cluster(p)] += 1

    return fvec


if __name__ == "__main__":

    import idc
    from sklearn.cluster import KMeans
    import random

    data = idc.IDCDataset(idc.PATIENTS, p=0.01)

    knndata = []
    for d in data.data:
        d = d.reshape(50, 50, 3)
        for i in range(10):
            x = random.randint(0, 49)
            y = random.randint(0, 49)
            knndata.append(d[x, y])

    kmeans = KMeans(n_clusters=20).fit(knndata)
    print(kmeans.cluster_centers_)
    print(kmeans.cluster_centers_ * 255)
