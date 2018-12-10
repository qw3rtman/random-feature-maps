
from sklearn.svm import LinearSVC

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


def run(ptrain=0.01, ptest=0.1, fdim=10000, ntrain=-25, ntest=25, n=20):

    timer = pinfo.Task("Random Fourier Feature Support Vector Classifier")
    rff = RandomFourierFeature(int(n), int(fdim))

    dataset = idc.IDCDataset(
        idc.PATIENTS[:int(ntrain)],
        p=float(ptrain), transform=rff.transform)
    test_dataset = idc.IDCDataset(
        idc.PATIENTS[-int(ntest):],
        p=float(ptest), transform=rff.transform)
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
