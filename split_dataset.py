from preprocess.dataset import split_dataset, save_dataset
import json
import argparse
import pickle
import os


def main(args):
    dataset_path = args.dataset_path
    val = args.val_percentage
    test = args.test_percentage

    if not os.path.exists(dataset_path):
        raise ValueError('dataset_path did\'t found')
    if not os.path.exists(dataset_path + '/cache/tokenizer.pkl'):
        raise ValueError(
            'tokenizer.pkl didn\'t found. please run create_tokenizer first')
    if not os.path.exists(dataset_path + '/captions.json'):
        raise ValueError('captions.json did\'t found')

    with open(dataset_path + '/cache/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open(dataset_path + '/captions.json', 'r') as f:
        caption_data = json.load(f)

    all_filenames = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
    all_captions = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
    for mode in ['happy', 'sad', 'angry']:
        for data in caption_data:
            if data['emotions'].get(mode):
                all_filenames[mode].append(data['filename'])
                all_captions[mode].append([data['emotions'][mode]])
    all_filenames['factual'] = [data['filename'] for data in caption_data]
    all_captions['factual'] = [[
        caption['edited'] for caption in data['captions']
    ] for data in caption_data]

    modes = ['happy', 'sad', 'angry']
    for mode in ['factual'] + modes:
        print(mode)
        train_indexes, val_indexes, test_indexes = split_dataset(
            all_filenames[mode], all_captions[mode], dataset_path + '/' + mode,
            tokenizer, val, test)
        print('train length', len(train_indexes))
        print('val length', len(val_indexes))
        print('test length', len(test_indexes))
        save_dataset(all_filenames[mode], all_captions[mode],
                     dataset_path + '/' + mode, train_indexes, val_indexes,
                     test_indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Splitter: Split dataset into train, val, and test set')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset/flickr10k',
        help='dataset folder path')
    parser.add_argument(
        '--val_percentage',
        type=int,
        default=10,
        help='validation for split percentage')
    parser.add_argument(
        '--test_percentage',
        type=int,
        default=10,
        help='test for split percentage')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
