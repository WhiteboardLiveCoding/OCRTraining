from os.path import basename, splitext

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    # Initialize data
    _, _, _, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            padding='valid',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(nb_filters,
                            kernel_size,
                            activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model
