
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


def run(ptrain=0.01, ptest=0.1):

    timer = pinfo.Task("Random Fourier Feature Support Vector Classifier")
    rff = RandomFourierFeature(1296, 10000)

    dataset = idc.IDCDataset(
        idc.PATIENTS[:-25], p=ptrain, transform=rff.transform)
    test_dataset = idc.IDCDataset(
        idc.PATIENTS[-25:], p=ptest, transform=rff.transform)
    tester = ClassifyTest(test_dataset.data, test_dataset.classes)

    rfsvm = train(dataset)
    tester.loss(rfsvm)

    debugtester = ClassifyTest(dataset.data, dataset.classes)
    debugtester.loss(rfsvm)

    timer.stop("Program finished.")


if __name__ == "__main__":
    import sys

    run(float(sys.argv[1]), float(sys.argv[2]))
