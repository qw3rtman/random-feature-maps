
"""Classification Tester

Attributes
----------
TRUTH : dict(str->int[])
    Definition of false positive, false negative, etc.
LABELS : dict(str->str)
    Human readable labels for various error statistics
"""

from syllabus import Task
import numpy as np


TRUTH = {
    'tn': [0, 0],
    'tp': [1, 1],
    'fn': [0, 1],
    'fp': [1, 0]
}

LABELS = {
    'total': 'Total Tests',
    'perr': 'Percent Error',
    'correct': 'Correct Tests',
    'incorrect': 'Incorrect Tests',
    'tn': 'True Negative',
    'tp': 'True Positive',
    'fn': 'False Negative',
    'fp': 'False Positive',
    'const_plus': 'Relative Improvement over Constant'
}


class ClassifyTest:
    """Binary Classifier Tester

    Parameters
    ----------
    data : np.array (shape=(N, D))
        Test data
    labels : np.array (shape=(N))
        Test labels (binary array)
    desc : str
        Tester description
    """

    def __init__(self, data, labels, desc=''):
        self.data = data
        self.labels = labels
        self.desc = desc

    def __error(self, predicted, truth):
        """Compute error

        Parameters
        ----------
        predicted : np.array
            Predicted classes
        truth : np.array
            True classes

        Returns
        -------
        dict
            'tn' : true negatives
            'tp' : true positives
            'fn' : false negatives
            'fp' : false positives
            'total' : total test cases
            'incorrect' : incorrect tests
            'correct' : correct tests
            'perr' : percent error
            'const_plus' : percent improvement over constant
        """

        error = {
            label: sum([
                1 for i, j in zip(predicted, truth)
                if (i == tn[0] and j == tn[1])])
            for label, tn in TRUTH.items()
        }

        total = predicted.shape[0]
        incorrect = np.linalg.norm(predicted - truth, ord=0)
        correct = total - incorrect
        perr = incorrect / total * 100
        const = min(sum(truth), total - sum(truth)) / total * 100

        error.update({
            'total': total,
            'incorrect': incorrect,
            'correct': correct,
            'perr': perr,
            'const_plus': (const - perr) / const
        })

        return error

    def loss(self, model, preprocess=None, task=None):
        """Run Loss Statistics

        Parameters
        ----------
        model : class with predict method
            Model to test
        preprocess : np.array -> np.array
            Preprocessing method to run; if None, no preprocessing is performed
        task : Task
            Task to register the test under
        """

        if task is not None:
            task = Task()
        task.reset()
        task.set_info(name='Classification Test', desc=self.desc)

        # Predict
        if preprocess is not None:
            predicted = model.predict(
                np.array([preprocess(x) for x in self.data]))
        else:
            predicted = model.predict(self.data)

        # Compute error
        self.error = self.__error(predicted, self.labels)

        # Print stats
        for key, value in self.error.items():
            task.info(
                "{label}: {val}".format(label=LABELS[key], val=value))

        task.done('Done running tests.')
        task.info(
            'Time per test: {t}s'
            .format(s=task.runtime() / self.error.total))
