from preprocess.images import process_images_all
from model import NIC
from keras import backend as K
import os
import json
import argparse


def assert_path_error(path):
    if not os.path.exists(path):
        raise ValueError(path + ' not found')


def main(args):
    dataset_path = args.dataset_path
    batch_size = args.batch_size

    assert_path_error(dataset_path)
    assert_path_error(dataset_path + '/captions.json')

    with open(dataset_path + '/captions.json', 'r') as f:
        caption_data = json.load(f)

    filenames = [data['filename'] for data in caption_data]

    encoder_resnet152 = NIC().model_encoder
    img_size = K.int_shape(encoder_resnet152.input)[1:3]
    transfer_values_size = K.int_shape(encoder_resnet152.output)[1]

    process_images_all(
        folder_path=dataset_path,
        filenames=filenames,
        img_size=img_size,
        transfer_values_size=transfer_values_size,
        image_model_transfer=encoder_resnet152,
        batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tokenizer: Create tokenizer object based on all dataset')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset/flickr10k',
        help='dataset folder path')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch size for processing image')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
