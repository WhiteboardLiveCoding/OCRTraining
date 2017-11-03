def load_data():
    import numpy as np
    from scipy.io import loadmat
    from keras.utils import np_utils

    emnist_file_path = 'dataset/emnist-byclass-fixed.mat'
    wlc_file_path = 'dataset/wlc-byclass-2017-10-23-22-20-27.197574.mat'

    height, width = 28, 28

    emnist = loadmat(emnist_file_path)
    wlc = None

    if wlc_file_path:
        wlc = loadmat(wlc_file_path)

    # Load training data
    training_images, training_labels = emnist['dataset'][0][0][0][0][0][0], emnist['dataset'][0][0][0][0][0][1]
    testing_images, testing_labels = emnist['dataset'][0][0][1][0][0][0], emnist['dataset'][0][0][1][0][0][1]

    if wlc_file_path:
        wlc_training_images, wlc_training_labels = wlc['dataset'][0][0][0][0][0][0], wlc['dataset'][0][0][0][0][0][1]
        wlc_testing_images, wlc_testing_labels = wlc['dataset'][0][0][1][0][0][0], wlc['dataset'][0][0][1][0][0][1]

        training_images = np.append(training_images, wlc_training_images, axis=0)
        training_labels = np.append(training_labels, wlc_training_labels, axis=0)

        testing_images = np.append(testing_images, wlc_testing_images, axis=0)
        testing_labels = np.append(testing_labels, wlc_testing_labels, axis=0)

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0], height, width, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], height, width, 1)

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    X_train = training_images
    X_test = testing_images

    Y_train = np_utils.to_categorical(training_labels, 94)
    Y_test = np_utils.to_categorical(testing_labels, 94)

    return X_train, Y_train, X_test, Y_test

