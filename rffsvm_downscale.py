
from sklearn.svm import LinearSVC

from skimage.transform import rescale
from skimage.color import hsv2rgb

import idc
from classify import ClassifyTest
from rff import RandomFourierFeature
import pinfo


def train(dataset):
    timer = pinfo.Task()
    rfsvm = LinearSVC()
    rfsvm.fit(dataset.data, dataset.classes)
    timer.stop("RFF SVC Computed", rfsvm)

    return rfsvm


def downscale(img):
    img = hsv2rgb(rescale(img, 0.5, anti_aliasing=False))
    return img.reshape([-1])


def run(ptrain=0.01, ptest=0.1, fdim=10000, ntrain=-25, ntest=25):

    timer = pinfo.Task("Random Fourier Feature Support Vector Classifier")
    rff = RandomFourierFeature(1875, int(fdim))

    dataset = idc.IDCDataset(
        idc.PATIENTS[:int(ntrain)],
        p=float(ptrain), feature=downscale, transform=rff.transform)
    test_dataset = idc.IDCDataset(
        idc.PATIENTS[-int(ntest):],
        p=float(ptest), feature=downscale, transform=rff.transform)
    tester = ClassifyTest(test_dataset.data, test_dataset.classes)

    rfsvm = train(dataset)
    tester.loss(rfsvm)
    tester.save()

    debugtester = ClassifyTest(dataset.data, dataset.classes)
    debugtester.loss(rfsvm)

    timer.stop("Program finished.")


if __name__ == "__main__":
    import sys
    from util import argparse
    args, kwargs = argparse(sys.argv[1:])
    run(**kwargs)
