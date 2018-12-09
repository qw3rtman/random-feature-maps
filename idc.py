
import os
import math
import random
import numpy as np
import matplotlib.image as mpimg
from skimage import feature

import pinfo

from multiprocessing import Pool
from functools import partial

BASE_PATH = os.path.join(os.getcwd(), 'IDC_regular_ps50_idx5')
PATIENTS = os.listdir(BASE_PATH)


class IDCDataset:

    def __init__(self, patients, transform=None, cores=None, p=1):
        self.transform = transform

        timer = pinfo.Task(label="Loading Images...")

        pool = Pool(cores)
        loaded = pool.map(partial(self.load_patient, p=p), patients)

        self.data = loaded[0][0]
        self.classes = loaded[0][1]
        for x in loaded[1:]:
            if x[0] is not None and x[1] is not None:
                try:
                    self.data = np.concatenate([self.data, x[0]])
                    self.classes = np.concatenate([self.classes, x[1]])
                except Exception as e:
                    print(x[0])
                    print(x[1])
                    pinfo.log(e)

        timer.stop(
            "{n} images ({p}%) sampled from {k} patients"
            .format(n=self.classes.shape[0], k=len(patients), p=p * 100),
            self.data, self.classes)

    def load_image(self, path):

        img = mpimg.imread(path)

        if img.shape[0] != 50 or img.shape[1] != 50 or img.shape[2] != 3:
            return None

        features = feature.hog(img, block_norm='L1', feature_vector=True)

        if features.shape[0] != 1296:
            return None
        try:
            return (
                features if self.transform is None
                else self.transform(features))
        except ValueError:
            return None

    def load_images(self, path, p=1):

        images = os.listdir(path)
        images = random.sample(images, math.ceil(len(images) * p))

        loaded = [self.load_image(os.path.join(path, img)) for img in images]
        return np.array([x for x in loaded if x is not None])

    def load_patient(self, patient, p=1):

        timer = pinfo.Task(tier=2)

        try:
            class_0 = self.load_images(
                os.path.join(BASE_PATH, patient, '0'), p=p)
            class_1 = self.load_images(
                os.path.join(BASE_PATH, patient, '1'), p=p)

            classes = np.concatenate([
                np.zeros(class_0.shape[0], dtype=np.int8),
                np.ones(class_1.shape[0], dtype=np.int8)])
            data = np.concatenate([class_0, class_1])

            timer.stop("loaded patient {p}".format(p=patient), data, classes)
            return (data, classes)

        except Exception as e:
            timer.stop(
                "error loading patient {p}: {e}".format(p=patient, e=e))
            return (None, None)
