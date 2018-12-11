
from print import *
# from sklearn.svm import LinearSVC
from syllabus import Task

from randomfeatures import RandomFourierFeature
from tester import IDCDataset, PATIENTS


if __name__ == "__main__":

    main = Task(name='MAIN')

    rff = RandomFourierFeature(7500, 10, kernel='G', task=main.subtask('RFF'))
    test = IDCDataset(
        PATIENTS[:5], task=main.subtask('load data'),
        p=0.001, transform=rff.transform)
