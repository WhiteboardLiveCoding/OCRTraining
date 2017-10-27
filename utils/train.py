import keras

from keras.models import save_model
from keras.utils import np_utils


def train(model, training_data, callback=True, batch_size=32, epochs=10):
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
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tb_callback] if callback else None)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
