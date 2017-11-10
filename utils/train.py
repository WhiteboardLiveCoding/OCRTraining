import keras
import tensorflow as tf

from keras.utils import np_utils

from utils.device import get_available_devices, normalize_device_name


def train(model, training_data, callback=True, batch_size=32, epochs=10, device='/cpu:0', parallel=False, verbose=1):

    if parallel:
        available_devices = get_available_devices()
        available_devices = [normalize_device_name(name) for name in available_devices]

        if device not in available_devices:
            raise ValueError('Target \'{}\' could not be found. Devices available are {}'.format(device,
                                                                                                 available_devices))

        score = _train_model(model, training_data, callback=callback,
                             batch_size=batch_size, epochs=epochs, verbose=verbose)
    else:
        with tf.device(device):
            score = _train_model(model, training_data, callback=callback,
                                 batch_size=batch_size, epochs=epochs, verbose=verbose)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def _train_model(model, training_data, callback=True, batch_size=32, epochs=10, verbose=1):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    if callback:
        # Callback for analysis in TensorBoard
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
                                                  write_images=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(x_test, y_test),
              callbacks=[tb_callback] if callback else None)

    return model.evaluate(x_test, y_test, verbose=0)
