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


def load_data(datasets):
    height, width = 28, 28

    dataset = loadmat(datasets[0])

    mapping = {kv[0]: kv[1:][0] for kv in dataset['dataset'][0][0][2]}

    # Load training data
    training_images, training_labels = _extract_images_and_labels(dataset)
    testing_images, testing_labels = _extract_images_and_labels(dataset, training_data=False)

    if len(datasets) > 0:
        for ds_path in datasets[1:]:
            ds = loadmat(ds_path)

            ds_training_images, ds_training_labels = _extract_images_and_labels(ds)
            ds_testing_images, ds_testing_labels = _extract_images_and_labels(ds, training_data=False)

            training_images = append_datasets(training_images, ds_training_images)
            training_labels = append_datasets(training_labels, ds_training_labels)

            testing_images = append_datasets(testing_images, ds_testing_images)
            testing_labels = append_datasets(testing_labels, ds_testing_labels)

    pickle.dump(mapping, open('bin/mapping.p', 'wb'))

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

