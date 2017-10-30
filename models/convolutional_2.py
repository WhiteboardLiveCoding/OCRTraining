from os.path import basename, splitext

from keras.layers import MaxPooling2D, Conv2D, Dropout, Dense, Flatten, Lambda, LeakyReLU, BatchNormalization
from keras.models import Sequential


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    _, _, mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(32, (3, 3)),
        LeakyReLU(),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Conv2D(64, (3, 3)),
        LeakyReLU(),
        BatchNormalization(axis=1),
        Conv2D(64, (3, 3)),
        LeakyReLU(),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.4),
        Dense(nb_classes, activation='softmax')
    ])

    return model
