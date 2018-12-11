
from print import *

from sklearn.svm import LinearSVC

from syllabus import Task

from randomfeatures import RandomFourierFeature
from tester import ClassifyTest, IDCDataset, PATIENTS


def train(dataset, task):

    task.reset()
    task.set_info(name="RFF SVC", desc="Computing RFF SVC Classifier")

    rfsvm = LinearSVC()
    rfsvm.fit(dataset.data, dataset.classes)

    task.done("RFF SVC Computed")

    return rfsvm


def run(
        ptrain=0.01, ptest=0.1,
        fdim=10000,
        ntrain=-25, ntest=25,
        n=20, knn=False,
        kernel="G"):

    div.div(
        '- -', BOLD,
        label='Random Fourier Feature Support Vector Classifier')
    table.table(list(map(list, zip(*[
        ['ptrain', ptrain],
        ['ptest', ptest],
        ['fdim', fdim],
        ['knn', knn],
        ['n', n],
        ['kernel', kernel],
        ['ntrain', ntrain],
        ['ntest', ntest]
    ]))))

    main = Task(
        name='RFF',
        desc="Random Fourier Feature Support Vector Classifier")

    rff = RandomFourierFeature(
        7500, 10,
        kernel='G', task=main.subtask("RFF"))

    params = {'p': float(ptrain), 'transform': rff.transform}
    """
    if knn:
        ckm = get_idc_colors(int(n))
        params['feature'] = ckm.map
    """
    # Load datasets
    dataset = IDCDataset(
        PATIENTS[:int(ntrain)],
        task=main.subtask("test data"), **params)
    test_dataset = IDCDataset(
        PATIENTS[-int(ntest):],
        task=main.subtask("training data"), **params)
    # Make tester
    tester = ClassifyTest(
        test_dataset.data, test_dataset.classes,
        'Classification experiment on new patients')
    # Debug tester
    debugtester = ClassifyTest(
        dataset.data, dataset.classes,
        'Classification verification on training data')
    """
    # Train model
    rfsvm = train(dataset)

    # Tester
    tester.loss(rfsvm, task=manager)

    # Debug tester
    debugtester.loss(rfsvm, task=manager)

    manager.done("Program finished.")
    """


if __name__ == "__main__":
    import sys
    from util import argparse
    args, kwargs = argparse(sys.argv[1:])
    run(**kwargs)
