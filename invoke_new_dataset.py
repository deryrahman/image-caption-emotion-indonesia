from preprocess.dataset import invoke_emotion_to_dataset, invoke_edited_to_dataset
import argparse
import os


def assert_path_error(path):
    if not os.path.exists(path):
        raise ValueError(path + ' not found')


def main(args):
    mongo_dump_path = args.mongo_dump_path
    dataset = args.dataset
    dataset_path = args.dataset_path

    assert_path_error(mongo_dump_path)
    assert_path_error(dataset_path)

    invoke_edited_to_dataset(mongo_dump_path, dataset_path)
    for mode in ['happy', 'sad', 'angry']:
        invoke_emotion_to_dataset(mongo_dump_path, dataset_path, dataset, mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Invoker: Invoke edited caption and emotion caption')

    parser.add_argument(
        '--mongo_dump_path',
        type=str,
        default='./dataset/dump/041619.json',
        help='mongo dump path')
    parser.add_argument(
        '--dataset',
        type=str,
        default='flickr',
        help='dataset either flickr or coco')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset/flickr10k',
        help='dataset folder path')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
