from preparation import split_dataset, save_dataset
from tokenizer import mark_captions, flatten, TokenizerWrap
import json
import argparse


def main(args):
    dataset_folder = args.dataset_folder
    val = args.val_percentage
    test = args.test_percentage

    with open(dataset_folder + '/captions.json', 'r') as f:
        caption_data = json.load(f)

    all_filenames = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
    all_captions = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
    for mode in ['happy', 'sad', 'angry']:
        for data in caption_data:
            if data['emotions'].get(mode):
                all_filenames[mode].append(data['filename'])
                all_captions[mode].append([data['emotions'][mode]])
    all_filenames['factual'] = [data['filename'] for data in caption_data]
    all_captions['factual'] = [
        [caption['id'] for caption in data['captions']] for data in caption_data
    ]

    modes = ['happy', 'sad', 'angry']
    captions_flat_all = []
    for mode in ['factual'] + modes:
        captions_marked = mark_captions(all_captions[mode])
        captions_flat = flatten(captions_marked)
        tokenizer = TokenizerWrap(texts=captions_flat)
        # remove oov words
        tmp = tokenizer.texts_to_sequences(captions_flat)
        captions_flat = tokenizer.sequences_to_texts(tmp)
        captions_flat_all.extend(captions_flat)
    tokenizer = TokenizerWrap(texts=captions_flat_all)

    modes = ['happy', 'sad', 'angry']
    for mode in ['factual'] + modes:
        train_indexes, val_indexes, test_indexes = split_dataset(
            all_filenames[mode], all_captions[mode],
            dataset_folder + '/' + mode, tokenizer, val, test)
        save_dataset(all_filenames[mode], all_captions[mode],
                     dataset_folder + '/' + mode, train_indexes, val_indexes,
                     test_indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Splitter: Split dataset into train, val, and test set')

    parser.add_argument(
        '--dataset_folder',
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
