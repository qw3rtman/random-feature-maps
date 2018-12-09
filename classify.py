
import pinfo
import numpy as np


class ClassifyTest:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def loss(self, model, preprocess=None):

        timer = pinfo.Task("Running test data:")

        if preprocess is not None:
            predicted = model.predict(
                np.array([preprocess(x) for x in self.data]))
        else:
            predicted = model.predict(self.data)

        incorrect = np.linalg.norm(predicted - self.labels, ord=0)
        print(
            "Incorrect labels: {i} ({perr:.1f}%)"
            .format(
                i=int(incorrect),
                perr=incorrect / self.labels.shape[0] * 100))
        print(
            "Correct labels: {i} ({pc:.1f})%"
            .format(
                i=int(self.labels.shape[0] - incorrect),
                pc=100 - incorrect / self.labels.shape[0] * 100))

        timer.stop("{n} tests run".format(n=self.labels.shape[0]))
