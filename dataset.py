from random import shuffle, choice

import argparse
import cv2
import numpy as np

from datetime import datetime

import sys
import os

from os.path import isfile, join
from scipy.io import loadmat, savemat
from math import floor

LOWEST_ALLOWED_CHAR = 33
HIGHEST_ALLOWED_CHAR = 126

MAX_ROTATION = 5
STEP = 1

TARGET_IMAGES = 1000
ADDITIONAL = [40, 41, 58, 61]

class Dataset:

    def __init__(self, batch_size=32):
        self._train_images = list()
        self._train_labels = list()

        self._test_images = list()
        self._test_labels = list()

        self.batch_size = batch_size
        self._load_dataset()

    def _load_dataset(self):
        self.data = loadmat('dataset/wlc-byclass.mat')

    def _append_to_dataset(self, test_data=False):
        if test_data:
            test_data = self.data['dataset'][0][0][1][0][0]
            self.data['dataset'][0][0][1][0][0][0] = np.append(test_data[0], self._test_images, axis=0)
            self.data['dataset'][0][0][1][0][0][1] = np.append(test_data[1], self._test_labels, axis=0)

            self._test_labels = list()
            self._test_images = list()
        else:
            train_data = self.data['dataset'][0][0][0][0][0]
            self.data['dataset'][0][0][0][0][0][0] = np.append(train_data[0], self._train_images, axis=0)
            self.data['dataset'][0][0][0][0][0][1] = np.append(train_data[1], self._train_labels, axis=0)

            self._train_labels = list()
            self._train_images = list()

    def add_image(self, image, label, test_data=False):
        if len(image) != len(self.data['dataset'][0][0][0][0][0][0][0]):
            raise Exception("Image data should be an array of length 784")

        reverse_mapping = {kv[1:][0]:kv[0] for kv in self.data['dataset'][0][0][2]}
        m_label = reverse_mapping.get(ord(label))

        if m_label is None:
            raise Exception("The dataset doesn't have a mapping for {}".format(label))

        if test_data:
            self._test_images.append(image)
            self._test_labels.append([m_label])
        else:
            self._train_images.append(image)
            self._train_labels.append([m_label])

        if len(self._test_images) >= self.batch_size or len(self._train_images) >= self.batch_size:
            self._append_to_dataset(test_data)

    def save(self, do_compression=True):
        if len(self._test_images) > 0:
            self._append_to_dataset(test_data=True)

        if len(self._train_images) > 0:
            self._append_to_dataset()

        file_name = 'dataset/wlc-byclass-{}.mat'.format(str(datetime.now()).replace(' ', '-').replace(':', '-'))
        savemat(file_name=file_name, mdict=self.data, do_compression=do_compression)

    def add_images_from_files(self, images, label, test_data):
        for img in images:
            self.add_image(img, label, test_data)


def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def normalize(img):
    img = np.reshape(img, 28 * 28)
    img = img.astype('float32')
    return img


def rotate_image(img, angle):
    # Calculate center, the pivot point of rotation
    (height, width) = img.shape[:2]
    center = (width // 2, height // 2)

    # Rotate
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rot_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img


def can_shift(img, i, j):
    shift = True

    if i == -1:
        shift = not np.any(img[0, :])
    elif i == 1:
        shift = not np.any(img[27, :])

    if j == -1 and shift:
        return not np.any(img[:, [0]])
    elif j == 1 and shift:
        return not np.any(img[:, [27]])
    return shift


def shift(img, i, j):
    top, bottom, left, right = 0, 0, 0, 0

    if i == -1:
        img = img[1:, :]
        bottom = 1
    elif i == 1:
        img = img[:27, :]
        top = 1

    if j == -1:
        img = img[:, 1:]
        right = 1
    elif j == 1 and shift:
        img = img[:, :27]
        left = 1

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def shift_image(img):
    images = list()
    for i in range(-1, 2):
        for j in range(-1, 2):
            if can_shift(img, i, j):
                shifted = shift(img, i, j)
                images.append(normalize(shifted))
    return images


def extend_image_set(images, count):
    extra = list()
    while len(images) + len(extra) < count:
        extra.append(choice(images))
    images.extend(extra)
    return images


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--images", type=str, help="Path to characters", required=True)
    parser.add_argument("-m", "--minimages", type=int, default=TARGET_IMAGES, help="Minimum number of characters")

    args, unknown = parser.parse_known_args()
    images = args.images
    minimages = args.minimages

    return images, minimages,


if __name__ == '__main__':
    images, min_images = arguments()

    dataset = Dataset()

    # for i in range(LOWEST_ALLOWED_CHAR, HIGHEST_ALLOWED_CHAR + 1):
    for i in ADDITIONAL:
        directory = '{}/{}'.format(images, i)
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
            images = list()

            for file in files:
                file_path = '{}/{}'.format(directory, file)
                img = cv2.imread(file_path)
                img = gray_scale(img)

                for angle in range(-MAX_ROTATION, MAX_ROTATION + STEP, STEP):
                    rotated = rotate_image(img, angle)
                    images.extend(shift_image(rotated))

            shuffle(images)
            training_count = floor(len(images) * 0.8)

            print('Character: {}, Set Length: {}'.format(chr(i), len(images)))
            training_set = extend_image_set(images[:training_count], round(min_images * 0.8))
            testing_set = extend_image_set(images[training_count:],  round(min_images * 0.2))

            dataset.add_images_from_files(training_set, chr(i), False)
            dataset.add_images_from_files(testing_set, chr(i), True)

    dataset.save()