
"""IDC Dataset Loader"""

import os
import math
import random
import numpy as np
import matplotlib.image as mpimg

from syllabus import Task

from functools import partial

BASE_PATH = os.path.join(os.getcwd(), 'IDC_regular_ps50_idx5')
PATIENTS = os.listdir(BASE_PATH)


def reducer(data, task=None):

    d = data[0][0]
    c = data[0][1]
    for i, j in data[1:]:
        try:
            d = np.concatenate([d, i])
            c = np.concatenate([c, j])
        except Exception as e:
            if task is not None:
                task.error('Failed to concatenate input data: ' + str(e))

    return (d, c)


class IDCDataset:
    """Invasive Ductile Carcinoma Dataset

    Parameters
    ----------
    patients : str[]
        List of patients to load
    feature : np.array->np.array
        Feature transform; if None, no transform is applied
    cores : int
        Number of cores to use
    p : float
        Proportion of data points to load
    task : Task
        Task object to register this object to
    """

    def __init__(
            self, patients,
            feature=None, transform=None,
            cores=None, p=1, task=None):

        self.transform = transform
        self.feature = feature

        if task is None:
            task = Task()
        task.reset(name='IDC Dataset', desc='Loading Images...')
        task.set_info(name='IDC Dataset', desc='Loading Images...')

        self.data, self.classes = task.pool(
            partial(self.load_patient, p=p), patients,
            reducer=reducer, name='Image Loader', recursive=False,
            cores=cores)

        task.done(
            "{n} images ({p}%) sampled from {k} patients"
            .format(n=self.classes.shape[0], k=len(patients), p=p * 100),
            self.data, self.classes)

    def load_image(self, path):
        """Load a single image

        Parameters
        ----------
        path : str
            Filepath to load the image from

        Returns
        -------
        np.array (shape=(D))
            One dimensional feature vector
        """

        img = mpimg.imread(path)

        if img.shape[0] != 50 or img.shape[1] != 50 or img.shape[2] != 3:
            return None

        if self.feature is not None:
            img = self.feature(img)
        else:
            img = img.reshape([-1])

        try:
            return img if self.transform is None else self.transform(img)
        except ValueError:
            return None

    def load_images(self, path, p=1):
        """Load images from a directory

        Parameters
        ----------
        path : str
            Filepath to load images from
        p : float
            Proportion of images to load (rounds up)

        Returns
        -------
        np.array (shape=(N,D))
            Images squished into a single vector, then concatenated
        """

        images = os.listdir(path)
        images = random.sample(images, math.ceil(len(images) * p))

        loaded = [self.load_image(os.path.join(path, img)) for img in images]
        return np.array([x for x in loaded if x is not None])

    def load_patient(self, patient, task=None, p=1):
        """Load a patient

        Parameters
        ----------
        patient : str
            patient to load
        task : Task
            task to register under
        """

        if task is None:
            task = Task()
        task.reset()
        task.set_info(
            name='Patient Loader',
            desc='Loading Patient {p}'.format(p=patient))
        task.info('Loading patient {p}...'.format(p=patient))

        try:
            class_0 = self.load_images(
                os.path.join(BASE_PATH, patient, '0'), p=p)
            class_1 = self.load_images(
                os.path.join(BASE_PATH, patient, '1'), p=p)

            classes = np.concatenate([
                np.zeros(class_0.shape[0], dtype=np.int8),
                np.ones(class_1.shape[0], dtype=np.int8)])
            data = np.concatenate([class_0, class_1])

            task.done("loaded patient {p}".format(p=patient), data, classes)
            return (data, classes)

        except Exception as e:
            task.error(
                "error loading patient {p}: {e}".format(p=patient, e=e))
            task.done("could not load patient {p}".format(p=patient))
            return (None, None)
