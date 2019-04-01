import json
import numpy as np


def convert_mongo(path):
    caption_flickr = []
    caption_coco = []
    # captions.json is a mongoexport from imagecaption.geekstudio.id
    with open(path) as f:
        content = f.readlines()
        for d in content:
            tmp = json.loads(d.rstrip())
            data = {
                'filename': tmp['file_name'],
                'captions': tmp['captions'],
                'emotions': {}
            }
            if tmp['image_id'].split('-')[0] == 'coco':
                caption_coco.append(data)
            else:
                caption_flickr.append(data)
    return caption_flickr, caption_coco


def split_and_save(captions, path_folder, seed=0, train_num=9000):
    n = len(captions)
    indexes = np.array([v['filename'].split('.')[0] for v in captions])
    np.random.seed(seed)
    np.random.shuffle(indexes)
    train_indexes = indexes[-train_num:,]
    validation_indexes = indexes[:n - train_num,]
    with open(path_folder + '/train.txt', 'w') as f:
        f.write('\n'.join(train_indexes))
    with open(path_folder + '/val.txt', 'w') as f:
        f.write('\n'.join(validation_indexes))


def load_caption(path_folder):
    filenames_train = ()
    captions_train = ()
    filenames_val = ()
    captions_val = ()
    with open(path_folder + '/caption.json', 'r') as f:
        data = json.load(f)
        with open(path_folder + '/train.txt', 'r') as f2:
            train_indexes = [d.rstrip() for d in f2.readlines()]
        with open(path_folder + '/val.txt', 'r') as f2:
            validation_indexes = [d.rstrip() for d in f2.readlines()]
        for d in data:
            idx = d['filename'].split('.')[0]
            filename = d['filename']
            if idx in train_indexes:
                filenames_train += (filename,)
                captions_train += ([cap['id'] for cap in d['captions']],)
            elif idx in validation_indexes:
                filenames_val += (filename,)
                captions_val += ([cap['id'] for cap in d['captions']],)
    return (filenames_train, captions_train), (filenames_val, captions_val)


if __name__ == '__main__':
    path = './dataset'
    flickr_folder = path + '/flickr10k'
    coco_folder = path + '/mscoco10k'

    # convert mongo to caption dict
    caption_flickr, caption_coco = convert_mongo(path=path + 'caption_bc.json')

    # save
    with open(flickr_folder + '/caption.json', 'w') as f:
        json.dump(caption_flickr, f)
    with open(coco_folder + '/caption.json', 'w') as f:
        json.dump(caption_coco, f)

    split_and_save(caption_flickr, flickr_folder)
    split_and_save(caption_coco, coco_folder)

    train, val = load_caption(flickr_folder)
    filenames_train, captions_train = train
    filenames_val, captions_val = val
