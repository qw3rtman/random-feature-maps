
"""Various helper functions for Random Features"""

from sklearn.svm import LinearSVC
from randomfeatures import RandomFourierFeature, RandomBinningFeature
from tester import ClassifyTest, IDCDataset, PATIENTS


def train(dataset, task):
    """Train a linear SVC on a transformed dataset

    Parameters
    ----------
    dataset : IDCDataset
        Dataset to train on
    task : Task
        Task to register the training under

    Returns
    -------
    LinearSVC
        Trained model
    """

    t = task.subtask(name="RF SVC", desc="Computing RF SVC Classifier")

    rfsvm = LinearSVC()
    rfsvm.fit(dataset.data, dataset.classes)

    t.done("RFF SVC Computed")

    return rfsvm


def make_feature(
        ftype='F', kernel='G',
        fdim=5000, idim=7500, task=None, cores=None):
    """Create a random feature

    Parameters
    -----------
    ftype : char
        Feature type
    kernel : char
        Kernel type
    fdim : int
        Number of features to generate
    idim : int
        Input space dimensionality
    task : Task
        Task to register under
    cores : int
        Number of cores to use
    """

    if ftype == 'F':
        return RandomFourierFeature(
            idim, fdim, kernel=kernel, task=task.subtask())
    elif ftype == 'B':
        return RandomBinningFeature(
            idim, fdim, task=task.subtask(), cores=cores)
    else:
        raise Exception("Unknown feature type {f}".format(f=ftype))


def make_datasets(
        transform, cores=None, feature=None,
        ntrain=-25, ntest=25, ptrain=0.01, ptest=0.1, main=None):
    """Create datasets and testers

    Parameters
    ----------
    transform : function (float[idim] -> float[fdim])
        Feature transform
    cores : int
        Number of processes to use
    feature : function (float[50][50][3] -> float[])
        Feature map; if None, no feature map is used
    ntrain : int
        Number of patients to train on
    ntest : int
        Number of patients to test on
    ptrain : float
        Proportion of training data to use
    ptest : float
        Proportion of test data to use
    main : Task
        Task to register dataset creation under
    """

    # Load datasets
    dataset = IDCDataset(
        PATIENTS[:ntrain], task=main.subtask("test data"),
        p=ptrain, cores=cores, transform=transform, feature=feature)
    test_dataset = IDCDataset(
        PATIENTS[-ntest:], task=main.subtask("training data"),
        p=ptest, cores=cores, transform=transform, feature=feature)
    # Make tester
    tester = ClassifyTest(
        test_dataset.data, test_dataset.classes,
        'Classification experiment on new patients')
    # Debug tester
    debugtester = ClassifyTest(
        dataset.data, dataset.classes,
        'Classification verification on training data')

    return [dataset, test_dataset, tester, debugtester]
