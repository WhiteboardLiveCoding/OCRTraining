import pickle
import numpy as np

from scipy.io import loadmat


def _rotate_image(img, width=28, height=28):
    """
    Used to rotate images read in from the EMNIST dataset
    """

    img.shape = (width, height)
    img = img.T
    img = list(img)
    img = [item for sublist in img for item in sublist]
    return img


def _extract_images_and_labels(dataset, training_data=True):
    if training_data:
        training_images = dataset['dataset'][0][0][0][0][0][0]
        training_labels = dataset['dataset'][0][0][0][0][0][1]
    else:
        training_images = dataset['dataset'][0][0][1][0][0][0]
        training_labels = dataset['dataset'][0][0][1][0][0][1]

    return training_images, training_labels


def append_datasets(arr1, arr2):
    return np.append(arr1, arr2, axis=0)


def load_data(emnist_file_path, wlc_file_path):
    height, width = 28, 28

    emnist = loadmat(emnist_file_path)
    wlc = None

    if wlc_file_path:
        wlc = loadmat(wlc_file_path)

    mapping = {kv[0]: kv[1:][0] for kv in emnist['dataset'][0][0][2]}

    if wlc_file_path:
        mapping = {kv[0]: kv[1:][0] for kv in wlc['dataset'][0][0][2]}

    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

    # Load training data
    training_images, training_labels = _extract_images_and_labels(emnist)
    testing_images, testing_labels = _extract_images_and_labels(emnist, training_data=False)

    # Reshape training data to be valid
    _len = len(training_images)
    for i in range(len(training_images)):
        print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        training_images[i] = _rotate_image(training_images[i])
    print('')

    # Reshape testing data to be valid
    _len = len(testing_images)
    for i in range(len(testing_images)):
        print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        testing_images[i] = _rotate_image(testing_images[i])
    print('')

    if wlc_file_path:
        wlc_training_images, wlc_training_labels = _extract_images_and_labels(wlc)
        wlc_testing_images, wlc_testing_labels = _extract_images_and_labels(wlc, training_data=False)

        training_images = append_datasets(training_images, wlc_training_images)
        training_labels = append_datasets(training_labels, wlc_training_labels)

        testing_images = append_datasets(testing_images, wlc_testing_images)
        testing_labels = append_datasets(testing_labels, wlc_testing_labels)

    # Extend the arrays to (None, 28, 28, 1)
    training_images = training_images.reshape(training_images.shape[0], height, width, 1)
    testing_images = testing_images.reshape(testing_images.shape[0], height, width, 1)

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    return (training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes

