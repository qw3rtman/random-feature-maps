
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


def run(train, test):

    timer = pinfo.Task("Random Fourier Feature Support Vector Classifier")
    rff = RandomFourierFeature(50 * 50 * 3, 7500)

    dataset = idc.IDCDataset(0, train, transform=rff.transform)
    test_dataset = idc.IDCDataset(train, train + test, transform=rff.transform)
    tester = ClassifyTest(test_dataset.data, test_dataset.classes)

    rfsvm = train(dataset)
    tester.loss(rfsvm)
    timer.stop("Program finished.")


if __name__ == "__main__":
    import sys

    run(int(sys.argv[1]), str(sys.argv[2]))
