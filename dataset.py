from random import shuffle

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
        print(len(image))
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

    def add_images_from_files(self, directory, files, label, test_data):
        for file in files:
            file_path = '{}/{}'.format(directory, file)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = np.reshape(img, 28 * 28)
            img = img.astype('float32')
            # Normalize to prevent issues with model
            img /= 255

            self.add_image(img, label, test_data)


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise ValueError('Expected 1 argument but got {}'.format(len(sys.argv) - 1))

    path = sys.argv[1]

    dataset = Dataset()

    for i in range(LOWEST_ALLOWED_CHAR, HIGHEST_ALLOWED_CHAR + 1):
        directory = '{}/{}'.format(path, i)
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
            shuffle(files)
            training_count = floor(len(files) * 0.8)

            training_set = files[:training_count]
            testing_set = files[training_count:]

            dataset.add_images_from_files(directory, training_set, chr(i), False)
            dataset.add_images_from_files(directory, testing_set, chr(i), True)

    dataset.save()