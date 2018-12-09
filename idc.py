
import os
import numpy as np
import matplotlib.image as mpimg

import pinfo

BASE_PATH = os.path.join(os.getcwd(), 'IDC_regular_ps50_idx5')
PATIENTS = os.listdir(BASE_PATH)


class IDCDataset:

    def __init__(self, start, end, transform=None):
        self.transform = transform
        self.data, self.classes = self.load(PATIENTS[start:end])

    def load_image(self, path):
        img = mpimg.imread(path).reshape([1, -1])[0]
        try:
            return img if self.transform is None else self.transform(img)
        except ValueError:
            return None

    def load_images(self, path):

        images = os.listdir(path)
        loaded = [self.load_image(os.path.join(path, img)) for img in images]

        return np.array(
            [i for i in loaded if i is not None and i.shape[0] == 50 * 50 * 3])

    def load_patient(self, patient):

        timer = pinfo.Task()

        class_0 = self.load_images(os.path.join(BASE_PATH, patient, '0'))
        class_1 = self.load_images(os.path.join(BASE_PATH, patient, '1'))
        classes = np.concatenate([
            np.zeros(class_0.shape[0], dtype=np.int8),
            np.ones(class_1.shape[0], dtype=np.int8)])
        data = np.concatenate([class_0, class_1])

        timer.stop("loaded patient {p}".format(p=patient), data, classes)

        return (data, classes)

    def load(self, patients):

        timer = pinfo.Task(label="Loading Images...")

        loaded = [self.load_patient(patient) for patient in patients]
        data = np.concatenate([x[0] for x in loaded])
        classes = np.concatenate([x[1] for x in loaded])

        timer.stop(
            "{n} images loaded from {k} patients"
            .format(n=classes.shape[0], k=len(patients)),
            data, classes)

        return (data, classes)
