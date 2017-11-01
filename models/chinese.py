from os.path import basename, splitext
import numpy as np

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    # Initialize data
    _, _, _, nb_classes = training_data
    kernel_size = (3, 3)
    pool_size = (2, 2)
    nb_f = width # nb_filters
    input_shape = (height, width, 1)

    model = Sequential()
    model.add(Conv2D(nb_f,
                     kernel_size=kernel_size,
                     weights=[np.random.normal(0, 0.01, size=(3, 3, 1, nb_f)), np.zeros(nb_f)],
                     activation='relu',
                     padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(nb_f * 2,
                     kernel_size=kernel_size,
                     weights=[np.random.normal(0, 0.01, size=(3, 3, nb_f, nb_f * 2)), np.zeros(nb_f * 2)],
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(nb_f * 4,
                     kernel_size=kernel_size,
                     weights=[np.random.normal(0, 0.01, size=(3, 3, nb_f * 2, nb_f * 4)), np.zeros(nb_f * 4)],
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(nb_f * 8, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
