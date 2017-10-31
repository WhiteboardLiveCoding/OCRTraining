from os.path import basename, splitext

from keras.layers import MaxPooling2D, Conv2D, Dropout, Dense, Flatten,\
    BatchNormalization, PReLU, ZeroPadding2D, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras.constraints import max_norm

MAX_NORM = 4  # Max-norm constraint on weights


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):

    _, _, _, nb_classes = training_data
    input_shape = (height, width, 1)

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    depth = nb_filters

    model = Sequential()
    model.add(Conv2D(
        depth,
        kernel_size=kernel_size,
        padding='same',
        kernel_constraint=max_norm(MAX_NORM),
        kernel_initializer='he_normal',
        input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))

    depth *= 2
    model.add(Conv2D(
        depth,
        kernel_size=kernel_size,
        kernel_initializer='he_normal',
        kernel_constraint=max_norm(MAX_NORM),
        padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    # to accomodate maxpooling on (15,15) shape
    # model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    depth *= 2
    model.add(Conv2D(
        depth,
        kernel_size=kernel_size,
        kernel_initializer='he_normal',
        kernel_constraint=max_norm(MAX_NORM),
        padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(2000, kernel_initializer='he_normal', kernel_constraint=max_norm(MAX_NORM)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2000, kernel_initializer='he_normal', kernel_constraint=max_norm(MAX_NORM)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, kernel_constraint=max_norm(MAX_NORM)))
    model.add(Activation('softmax'))

    return model

