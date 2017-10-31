from os.path import basename, splitext
import numpy as np

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    # Initialize data
    _, _, _, nb_classes = training_data
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Conv2D(width, (3, 3), weights=[np.random.normal(0, 0.01, size=(3, 3, 1, width)), np.zeros(width)],
                     activation='relu', padding='same', strides=(1, 1),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(width * 2, (3, 3), weights=[np.random.normal(0, 0.01, size=(3, 3, width, width * 2)), np.zeros(width * 2)],
                     activation='relu', padding='same', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(width * 4, (3, 3), weights=[np.random.normal(0, 0.01, size=(3, 3, width * 2, width * 4)), np.zeros(width * 4)],
                     activation='relu', padding='same', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(width * 8, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
