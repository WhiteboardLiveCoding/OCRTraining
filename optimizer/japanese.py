from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from keras.layers.core import Dense, Dropout

from optimizer.dataset import load_data


def model(X_train, Y_train, X_test, Y_test):
    nb_classes = 94
    input_shape = (28, 28, 1)

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, kernel_size=kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Conv2D(nb_filters * 2, kernel_size=kernel_size))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters * 2, kernel_size=kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data()

    best_run, best_model = optim.minimize(model=model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))