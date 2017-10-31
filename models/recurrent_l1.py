from os.path import basename, splitext

from keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Input, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model


def get_model_id():
    return splitext(basename(__file__))[0]


def build(training_data, height=28, width=28):
    _, _, mapping, nb_classes = training_data
    # input_shape = (height, width, 1)

    nb_filters = 128  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    input_img = Input(shape=(height, width, 1))

    conv_l = Conv2D(nb_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu')
    l = conv_l(input_img)

    conv1 = Conv2D(nb_filters,
                   kernel_size=(1, 1),
                   padding='same')
    stack1 = conv1(l)
    stack2 = BatchNormalization()(stack1)
    stack3 = PReLU()(stack2)

    # conv2 = Conv2D(nb_filters,
    #                kernel_size=kernel_size,
    #                padding='same',
    #                kernel_initializer='he_normal')
    # stack4 = conv2(stack3)
    # stack5 = add([stack1, stack4])
    # stack6 = BatchNormalization()(stack5)
    # stack7 = PReLU()(stack6)

    # conv3 = Conv2D(nb_filters,
    #                kernel_size=kernel_size,
    #                padding='same',
    #                weights=conv2.get_weights())
    # stack8 = conv3(stack7)
    # stack9 = add([stack1, stack8])
    # stack10 = BatchNormalization()(stack9)
    # stack11 = PReLU()(stack10)
    #
    # conv4 = Conv2D(nb_filters,
    #                kernel_size=kernel_size,
    #                padding='same',
    #                weights=conv3.get_weights())
    # stack12 = conv4(stack11)
    # stack13 = add([stack1, stack12])
    # stack14 = BatchNormalization()(stack13)
    # stack15 = PReLU()(stack14)

    stack16 = MaxPooling2D(pool_size, padding='same')(stack3)
    stack17 = Dropout(0.1)(stack16)

    out = Flatten()(stack17)

    l_out = Dense(nb_classes, activation='softmax')(out)

    model = Model(inputs=input_img, outputs=l_out)

    return model
