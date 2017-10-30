import argparse

from scipy.io import loadmat, savemat


def _rotate_image(img, width=28, height=28):
    img.shape = (width, height)
    img = img.T
    img = list(img)
    img = [item for sublist in img for item in sublist]
    return img


def fix_and_save_dataset(emnist_file_path):
    dataset = loadmat(emnist_file_path)

    # Reshape training data to be valid
    _len = len(dataset['dataset'][0][0][0][0][0][0])
    for i in range(len(dataset['dataset'][0][0][0][0][0][0])):
        print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        dataset['dataset'][0][0][0][0][0][0][i] = _rotate_image(dataset['dataset'][0][0][0][0][0][0][i])
    print('')

    # Reshape testing data to be valid
    _len = len(dataset['dataset'][0][0][1][0][0][0])
    for i in range(len(dataset['dataset'][0][0][1][0][0][0])):
        print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1) / _len) * 100), end='\r')
        dataset['dataset'][0][0][1][0][0][0][i] = _rotate_image(dataset['dataset'][0][0][1][0][0][0][i])
    print('')

    savemat(file_name='dataset/emnist-byclass-fixed.mat', mdict=dataset, do_compression=True)


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--emnist", type=str, help="Path to the emnist(by class) dataset", required=True)

    args, unknown = parser.parse_known_args()
    emnist = args.emnist

    return emnist


if __name__ == '__main__':
   emnist = arguments()
   fix_and_save_dataset(emnist)
