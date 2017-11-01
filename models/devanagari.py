from os.path import basename, splitext

from keras.layers import MaxPooling2D, Conv2D, Dropout, Dense, Flatten, Activation
from keras.models import Sequential


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    # Initialize data
    _, _, _, nb_classes = training_data
    kernel_size = (3, 3)
    pool_size = (2, 2)
    input_shape = (height, width, 1)

    model = Sequential()

    model.add(Conv2D(100,
                     kernel_size=kernel_size,
                     padding='valid',
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Activation('tanh'))

    model.add(Conv2D(250,
                     kernel_size=kernel_size,
                     padding='valid'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Activation('tanh'))

    model.add(Flatten())

    model.add(Dense(1000))

    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Dense(nb_classes))

    model.add(Activation('softmax'))

    return model
