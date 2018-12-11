
from print import *

from sklearn.svm import LinearSVC

from syllabus import Task

from randomfeatures import RandomFourierFeature, RandomBinningFeature, CKM
from tester import ClassifyTest, IDCDataset, PATIENTS
import multiprocessing


def get_idc_colors(k):
    """Estimate the k-best colors for the IDC dataset by sampling 1% of the
    dataset.

    Parameters
    ----------
    k : int
        Number of colors

    Returns
    -------
    CKM
        Color K Means classifier
    """
    return CKM(k, IDCDataset(PATIENTS, p=0.01))


def train(dataset, task):

    task.reset()
    task.set_info(name="RF SVC", desc="Computing RF SVC Classifier")

    rfsvm = LinearSVC()
    rfsvm.fit(dataset.data, dataset.classes)

    task.done("RFF SVC Computed")

    return rfsvm


def run(
        ptrain=0.01, ptest=0.1,
        fdim=5000,
        ntrain=-25, ntest=25,
        n=20, knn=False,
        kernel="G", cores=None, ftype='F'):

    div.div(
        '- -', BOLD,
        label='Random Feature Support Vector Classifier')

    print('\n')
    div.div('-', BR + BLUE, label='Parameters')
    params = [
        ['ptrain', ptrain, 'Percent of training images to Use'],
        ['ptest', ptest, 'Percent of test images to Use'],
        ['fdim', fdim, 'Feature space dimensionality'],
        ['knn', knn, 'Use Color K Means?'],
        ['n', n, 'Number of means'],
        ['kernel', kernel, 'Kernel Type ("R", "L", or "C")'],
        ['ntrain', ntrain, 'Number of patients used for training'],
        ['ntest', ntest, 'Number of patients used for testing'],
        [
            'cores', cores,
            'Number of processes (cores) to use ({n} available)'
            .format(n=multiprocessing.cpu_count())],
        ['ftype', ftype, 'type of feature to use ("F" or "B")']
    ]
    table.table(params, padding=' ' * 4)
    print('\n\n')

    div.div('-', BR + BLUE, label='Main Program')

    main = Task(
        name='RF',
        desc="Random Feature Support Vector Classifier")

    if ftype == 'F':
        rf = RandomFourierFeature(
            7500 if bool(knn) is False else int(n), int(fdim),
            kernel='G', task=main.subtask("RFF"))
    elif ftype == 'B':
        rf = RandomBinningFeature(
            7500 if bool(knn) is False else int(n), int(fdim),
            task=main.subtask("RFF"),
            cores=None if cores is None else int(cores))
    else:
        raise Exception("Unknown feature type")

    params = {
        'p': float(ptrain),
        'transform': rf.transform,
        'cores': None if cores is None else int(cores)
    }

    if knn:
        ckm = get_idc_colors(int(n))
        params['feature'] = ckm.map

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

    # Train model
    rfsvm = train(dataset, main.subtask())

    # Tester
    tester.loss(rfsvm, task=main)

    # Debug tester
    debugtester.loss(rfsvm, task=main)

    main.done("Program finished.")


if __name__ == "__main__":
    import sys
    from util import argparse

    putil.LOG_FILE = 'results_' + '_'.join(sys.argv[1:]) + '.txt'
    args, kwargs = argparse(sys.argv[1:])
    run(**kwargs)
