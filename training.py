# Mute tensorflow debugging information console
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(usage='A training script for the Live Whiteboard Coding neural network')
    parser.add_argument('--emnist', type=str, help='path for the EMNIST dataset', required=True)
    parser.add_argument('--wlc', type=str, help='path for the WLC dataset')
    parser.add_argument('-o', '--output', type=str, help='output directory for the model(without /)', default='bin')
    parser.add_argument('--height', type=int, default=28, help='height of the input image')
    parser.add_argument('--width', type=int, default=28, help='width of the input image')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-g', '--gpus', type=int, default=1, help='number of gpus to be used')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size for training')
    parser.add_argument('-d', '--device', type=str, default='/cpu:0', help='device to be used for training')
    parser.add_argument('-m', '--model', type=str, default='convolutional', help='keras model to be trained')
    parser.add_argument('-p', '--parallel', action='store_true', default=False, help='use multi gpu model')

    return parser.parse_args()


def main():
    if args.output[0] is '/':
        print('Please make sure that the output directory has no leading \'/\'')
        sys.exit(1)

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/' + args.output
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    training_data = load_data(args.emnist, args.wlc)

    model = build_model(training_data=training_data,
                        model_id=args.model,
                        height=args.height,
                        width=args.width,
                        multi_gpu=args.parallel,
                        gpus=args.gpus)

    if not model:
        print('Model {} does not exist.'.format(args.model))
        sys.exit(1)

    train(model, training_data, epochs=args.epochs, batch_size=args.batch, device=args.device)

    save_model_to_file(model, args.output)


if __name__ == '__main__':
    args = parse_arguments()

    from utils.dataset import load_data
    from utils.model import build_model, save_model_to_file
    from utils.train import train

    main()
