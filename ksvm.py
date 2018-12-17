
"""Kernel SVM Baseline Test"""

from print import *
from syllabus import Task
from helpers import make_trainset, make_testset
from sklearn.svm import SVC


LOG_FILE_DIR = 'results'


def train(dataset, task):
    task.start()
    ksvm = SVC()
    ksvm.fit(dataset.data, dataset.classes)
    task.done(desc="Kernel SVM Computed")

    return ksvm


def run(ptrain=0.01, ptest=0.1, ntrain=-25, ntest=25):

    import os
    putil.LOG_FILE = os.path.join(LOG_FILE_DIR, 'ksvm.txt')

    main = Task(
        name="Kernel SVM",
        desc='Kernel Support Vector Machine Baseline').start()

    dataset, validation_tester = make_trainset(
        cores=None, main=main, ptrain=float(ptrain))
    testset, tester = make_testset(
        cores=None, main=main, ptest=float(ptest))

    ksvm = train(dataset, main.subtask(name="Training KSVM"))

    tester.loss(ksvm, task=main)
    validation_tester.loss(ksvm, task=main)

    main.done("Program finished.")


if __name__ == "__main__":
    run(ptrain=0.2, ptest=1, ntrain=-25, ntest=25)
