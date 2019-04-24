from preprocess.tokenizer import mark_captions, flatten, TokenizerWrap
import pickle
import os
import argparse
import json


def assert_path_error(path):
    if not os.path.exists(path):
        raise ValueError(path + ' not found')


def main(args):
    dataset_path = args.dataset_path
    num_words = args.num_words
    if num_words < 0:
        num_words = None
    assert_path_error(dataset_path)
    assert_path_error(dataset_path + '/captions.json')

    with open(dataset_path + '/captions.json', 'r') as f:
        caption_data = json.load(f)

    try:
        captions = [[data['emotions'][mode]]
                    for mode in ['happy', 'sad', 'angry']
                    for data in caption_data
                    if data['emotions'].get(mode)]
        captions += [[caption['edited']
                      for caption in data['captions']]
                     for data in caption_data]
    except Exception as e:
        print(e)
        print('captions.json structure is not proper,'
              ' please run invoke_new_dataset first')
        raise e

    captions_marked = mark_captions(captions)
    captions_flat = flatten(captions_marked)
    tokenizer = TokenizerWrap(texts=captions_flat, num_words=num_words)
    print('num_words', num_words)

    if not os.path.exists(dataset_path + '/cache'):
        os.mkdir(dataset_path + '/cache')

    with open(dataset_path + '/cache/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('save tokenizer complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tokenizer: Create tokenizer object based on all dataset')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset/flickr10k',
        help='dataset folder path')
    parser.add_argument(
        '--num_words',
        type=int,
        default=10000,
        help=('num of words tokenizer,'
              'if negative, then use all words within dataset'))

    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
