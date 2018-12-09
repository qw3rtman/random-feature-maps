
from sklearn.svm import SVC

import idc
from classify import ClassifyTest
import pinfo


def train(dataset):
    timer = pinfo.Task()
    ksvm = SVC()
    ksvm.fit(dataset.data, dataset.classes)
    timer.stop("Kernel SVM Computed", ksvm)

    return ksvm


def run(ptrain=0.01, ptest=0.1):

    timer = pinfo.Task("Kernel SVM")

    dataset = idc.IDCDataset(idc.PATIENTS[:-25], p=ptrain)
    test_dataset = idc.IDCDataset(idc.PATIENTS[-25:], p=ptest)
    tester = ClassifyTest(test_dataset.data, test_dataset.classes)

    ksvm = train(dataset)
    tester.loss(ksvm)

    debugtester = ClassifyTest(dataset.data, dataset.classes)
    debugtester.loss(ksvm)

    timer.stop("Program finished.")


if __name__ == "__main__":
    import sys
    from util import argparse
    args, kwargs = argparse(sys.argv[1:])
    run(ptrain=argparse["ptrain"], ptest=argparse["ptest"])
