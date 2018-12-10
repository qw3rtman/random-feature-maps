
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
        pinfo.log(
            "Incorrect labels: {i} ({perr:.1f}%)"
            .format(
                i=int(incorrect),
                perr=incorrect / self.labels.shape[0] * 100))
        pinfo.log(
            "Correct labels: {i} ({pc:.1f})%"
            .format(
                i=int(self.labels.shape[0] - incorrect),
                pc=100 - incorrect / self.labels.shape[0] * 100))

        fp, fn, tp, tn = [0, 0, 0, 0]
        for i, j in zip(predicted, self.labels):
            if i == 0 and j == 0:
                tn += 1
            if i == 1 and j == 1:
                tp += 1
            if i == 0 and j == 1:
                fn += 1
            if i == 1 and j == 0:
                fp += 1
        pinfo.log(
            "False positive: {i}".format(i=fp))
        pinfo.log(
            "False negative: {i}".format(i=fn))
        pinfo.log(
            "True positive: {i}".format(i=tp))
        pinfo.log(
            "True negative: {i}".format(i=tn))

        timer.stop("{n} tests run".format(n=self.labels.shape[0]))

        with open('results.txt', 'w') as file:
            file.writelines(str(fp), str(fn), str(tp), str(tn))
