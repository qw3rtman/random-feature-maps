
"""IDC Dataset Loader"""

import os
import math
import random
import numpy as np
import matplotlib.image as mpimg

from syllabus import Task


BASE_PATH = os.path.join(os.getcwd(), 'IDC_regular_ps50_idx5')
PATIENTS = os.listdir(BASE_PATH)
PATIENTS.sort()


def load_image(path, feature=None, transform=None):
    """Load a single image

    Parameters
    ----------
    path : str
        Filepath to load the image from
    feature : np.array (shape=(50,50,3)) -> np.array (shape=(-1)) or None
        Feature map to use; if None, the feature map is computed by reshaping
        the array into a vector
    transform : np.array (shape=(-1)) -> np.array (shape=(-1)) or None
        Image transformation to apply after the feature map

    Returns
    -------
    np.array (shape=(D))
        One dimensional feature vector
    """

    img = mpimg.imread(path)

    if img.shape[0] != 50 or img.shape[1] != 50 or img.shape[2] != 3:
        return None

    if feature is not None:
        img = feature(img)
    else:
        img = img.reshape([-1])

    try:
        return img if transform is None else transform(img)
    except ValueError:
        return None


def load_images(path, p=1, feature=None, transform=None):
    """Load images from a directory

    Parameters
    ----------
    path : str
        Filepath to load images from
    p : float
        Proportion of images to load (rounds up)
    feature : np.array (shape=(50,50,3)) -> np.array (shape=(-1)) or None
        Feature map to use; if None, the feature map is computed by reshaping
        the array into a vector
    transform : np.array (shape=(-1)) -> np.array (shape=(-1)) or None
        Image transformation to apply after the feature map

    Returns
    -------
    np.array (shape=(N,D))
        Images squished into a single vector, then concatenated
    """

    images = os.listdir(path)
    images = random.sample(images, math.ceil(len(images) * p))

    loaded = [
        load_image(
            os.path.join(path, img),
            feature=feature, transform=transform)
        for img in images]

    return np.array([x for x in loaded if x is not None])


def load_patient(patient, p=1, transform=None, feature=None, task=None):
    """Load a patient

    Parameters
    ----------
    args : [str, Task, array]
        [0] patient to load
        [1] task to register under
        [2] [float, class, mixed type[], function]
            [0] proportion of samples to load
            [1] feature generator class
            [2] feature arguments
            [3] image transform
    """

    # Set up task
    if task is None:
        task = Task()
    task.start(name='Loader', desc='Loading Patient {p}'.format(p=patient))

    # Config: pass in p, feature, transform
    load_args = {
        'p': p,
        'feature': feature,
        'transform': transform
    }

    try:
        class_0 = load_images(
            os.path.join(BASE_PATH, patient, '0'), **load_args)
        class_1 = load_images(
            os.path.join(BASE_PATH, patient, '1'), **load_args)

        classes = np.concatenate([
            np.zeros(class_0.shape[0], dtype=np.int8),
            np.ones(class_1.shape[0], dtype=np.int8)])
        data = np.concatenate([class_0, class_1])

        task.done(data, classes, desc="loaded patient {p}".format(p=patient))
        return (data, classes)

    except Exception as e:
        task.error(
            "error loading patient {p}: {e}".format(p=patient, e=e))
        task.done(desc="could not load patient {p}".format(p=patient))
        return (None, None)


SHARED_ARGS = {}


def pinit(tgen, targs, p, feature):

    global SHARED_ARGS
    if tgen is not None and targs is not None:
        SHARED_ARGS['transform'] = tgen(*targs).transform
    else:
        SHARED_ARGS['transform'] = None
    SHARED_ARGS['p'] = p
    SHARED_ARGS['feature'] = feature


def proc_wrapper(args):

    patient, task = args
    global SHARED_ARGS
    return load_patient(
        patient, p=SHARED_ARGS['p'],
        transform=SHARED_ARGS['transform'],
        feature=SHARED_ARGS['feature'],
        task=task)


def proc_reducer(args):
    data, task = args
    return reducer(data, task)


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
    feature : np.array -> np.array
        Feature transform; if None, no transform is applied
    transform : np.array -> np.array or [tgen, targs]
        Transformation to apply. This should be a function in threading
        mode; in process mode, this should be a tuple.
        [0] : Class
            generator class for the feature transform
        [1] : T[]
            list of args to pass to fgen
    cores : int
        Number of cores to use
    p : float
        Proportion of data points to load
    task : Task
        Task object to register this object to
    """

    def __init__(
            self, patients,
            transform=[None, None], feature=None,
            cores=None, p=1, task=None, process=False):

        if task is None:
            task = Task()
        task.start(name='IDC Dataset', desc='Loading Images...')

        if transform is None:
            transform = [None, None]

        if process:
            self.data, self.classes = task.pool(
                proc_wrapper, patients,
                process=True,
                shared_args=[transform[0], transform[1], p, feature],
                shared_init=pinit,
                name='Loader',
                recursive=True,
                reducer=proc_reducer,
                cores=cores)

        else:
            self.data, self.classes = task.pool(
                load_patient, patients,
                process=False,
                shared_kwargs={
                    'transform': transform,
                    'feature': feature,
                    'p': p},
                reducer=reducer,
                name='Loader',
                recursive=False,
                threads=cores)

        task.done(
            self.data, self.classes,
            desc="{n} images ({p}%) sampled from {k} patients".format(
                n=self.classes.shape[0], k=len(patients), p=p * 100))
