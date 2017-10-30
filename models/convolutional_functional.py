from os.path import basename, splitext

from keras.layers import MaxPooling2D, Conv2D, Dropout, Dense, Flatten, Input
from keras.models import Model


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    # Initialize data
    _, _, mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    # Hyperparameters
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    input_img = Input(shape=input_shape)

    conv_1 = Conv2D(nb_filters,
                    kernel_size,
                    padding='valid',
                    input_shape=input_shape,
                    activation='relu')(input_img)

    conv_2 = Conv2D(nb_filters,
                    kernel_size,
                    activation='relu')(conv_1)

    max_pooling = MaxPooling2D(pool_size=pool_size)(conv_2)

    dropout_1 = Dropout(0.25)(max_pooling)

    flatten_1 = Flatten()(dropout_1)

    dense_1 = Dense(512, activation='relu')(flatten_1)

    dropout_2 = Dropout(0.5)(dense_1)

    out = Dense(nb_classes, activation='softmax')(dropout_2)

    model = Model(inputs=input_img, outputs=out)

    return model
