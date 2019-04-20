from preprocess.dataset import invoke_emotion_to_dataset, invoke_edited_to_dataset
import argparse


def main(args):
    mongo_dump_path = args.mongo_dump_path
    dataset = args.dataset
    dataset_folder = args.dataset_folder
    invoke_edited_to_dataset(mongo_dump_path, dataset_folder)
    for mode in ['happy', 'sad', 'angry']:
        invoke_emotion_to_dataset(mongo_dump_path, dataset_folder, dataset,
                                  mode)


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
        '--dataset_folder',
        type=str,
        default='./dataset/flickr10k',
        help='dataset folder path')

    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
