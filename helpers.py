
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

    t = task.start(name="RF SVC", desc="Computing RF SVC Classifier")

    rfsvm = LinearSVC()
    rfsvm.fit(dataset.data, dataset.classes)

    t.done(desc="RFF SVC Computed")

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

    Returns
    -------
    (class, mixed type arr)
        [0] Feature generator used
        [1] MP-ready packaged parameters
    """

    if ftype == 'F':
        return (
            RandomFourierFeature,
            RandomFourierFeature(
                idim, fdim, kernel=kernel, task=task.subtask()).mp_package())
    elif ftype == 'B':
        return (
            RandomBinningFeature,
            RandomBinningFeature(
                idim, fdim, cores=cores, task=task.subtask()).mp_package())
    else:
        raise Exception("Unknown feature type {f}".format(f=ftype))


def make_trainset(
        cores=None, feature=None, transform=None,
        ntrain=-25, ptrain=0.01, main=None):
    """Create training dataset and validiation tester

    Parameters
    ----------
    cores : int
        Number of processes to use
    feature : function (float[50][50][3] -> float[])
        Feature map; if None, no feature map is used
    ntrain : int
        Number of patients to train on
    ptrain : float
        Proportion of training data to use
    main : Task
        Task to register dataset creation under
    """

    # Load dataset
    main.print("Loading Training Data:")
    dataset = IDCDataset(
        PATIENTS[:ntrain], cores=cores, feature=feature, process=True,
        task=main.subtask(), p=ptrain, transform=transform)

    # debug tester
    debugtester = ClassifyTest(
        dataset.data, dataset.classes,
        'Classification verification on training data')

    return dataset, debugtester


def make_testset(
        cores=None, feature=None, transform=None,
        ntest=25, ptest=0.1, main=None):
    """Create testing dataset and tester

    Parameters
    ----------
    cores : int
        Number of processes to use
    feature : function (float[50][50][3] -> float[])
        Feature map; if None, no feature map is used
    ntest : int
        Number of patients to test on
    ptest : float
        Proportion of test data to use
    main : Task
        Task to register dataset creation under
    """

    # Load dataset
    main.print("Loading Testing Data:")
    test_dataset = IDCDataset(
        PATIENTS[-ntest:], transform=transform, cores=cores, feature=feature,
        task=main.subtask(), p=ptest, process=True)
    # Make tester
    tester = ClassifyTest(
        test_dataset.data, test_dataset.classes,
        'Classification experiment on new patients')

    return test_dataset, tester
