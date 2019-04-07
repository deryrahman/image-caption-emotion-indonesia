import json
import numpy as np
import pickle
import os


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


def load_caption(path_folder, lang='id', modes=['happy', 'sad', 'angry']):
    filenames_train = {mode: () for mode in modes}
    filenames_train.update({'factual': ()})
    captions_train = {mode: () for mode in modes}
    captions_train.update({'factual': ()})
    filenames_val = {mode: () for mode in modes}
    filenames_val.update({'factual': ()})
    captions_val = {mode: () for mode in modes}
    captions_val.update({'factual': ()})
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
                filenames_train['factual'] += (filename,)
                captions_train['factual'] += ([
                    cap[lang] for cap in d['captions']
                ],)
                for mode in modes:
                    caption_with_mode = d['emotions'].get(mode)
                    if caption_with_mode is not None:
                        filenames_train[mode] += (filename,)
                        captions_train[mode] += ([caption_with_mode],)
            elif idx in validation_indexes:
                filenames_val['factual'] += (filename,)
                captions_val['factual'] += ([
                    cap[lang] for cap in d['captions']
                ],)
                for mode in modes:
                    caption_with_mode = d['emotions'].get(mode)
                    if caption_with_mode is not None:
                        filenames_val[mode] += (filename,)
                        captions_val[mode] += ([caption_with_mode],)
    return (filenames_train, captions_train), (filenames_val, captions_val)


def process_caption_index_helper(path, flickr_folder, coco_folder, save_path):
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as f:
            image_id_to_idx = pickle.load(f)
        return image_id_to_idx
    image_id_to_filename = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            image_id_to_filename[data['image_id']] = data['file_name']
    filename_to_idx = {}
    for folder in [flickr_folder, coco_folder]:
        with open(folder + '/caption.json', 'r') as f:
            caption = json.load(f)
            for i, r in enumerate(caption):
                filename_to_idx[r['filename']] = i
    image_id_to_idx = {}
    for image_id, filename in image_id_to_filename.items():
        image_id_to_idx[image_id] = filename_to_idx[filename]
    with open(save_path, 'wb') as f:
        pickle.dump(image_id_to_idx, f)
    return image_id_to_idx


def invoke_emotion_to_dataset(mongo_dump_path, image_id_to_idx, dataset_folder,
                              dataset, emotion):
    if dataset != 'flickr' and dataset != 'coco':
        raise ValueError('dataset only flickr or coco')
    with open(dataset_folder + '/caption.json', 'r') as f:
        caption = json.load(f)
    with open(mongo_dump_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            user = json.loads(line)
            for user_caption in user['captions']:
                if user_caption['step'] != 'emotion' or user_caption[
                        'captionEmotion'][emotion] == '':
                    continue
                if user_caption['image_id'].split('-')[0] != dataset:
                    continue
                caption[image_id_to_idx[user_caption['image_id']]]['emotions'][
                    emotion] = user_caption['captionEmotion'][emotion]
    with open(dataset_folder + '/caption.json', 'w') as f:
        json.dump(caption, f)
    return caption


def invoke_edited_to_dataset(mongo_dump_path, dataset_folder):
    mp = {}
    with open(mongo_dump_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            test = json.loads(line)
            for cap in test['captions']:
                if len(cap['captionEdit']) == 0:
                    continue
                for tmp in cap['captionEdit']:
                    caption_id = tmp['caption_id']
                    if mp.get(caption_id) is None:
                        mp[caption_id] = []
                    mp[caption_id].append(tmp['text'])
    with open(dataset_folder + '/caption.json', 'r') as f:
        data = json.load(f)
    for j in range(len(data)):
        for i, tt in enumerate(data[j]['captions']):
            caption_id = tt['caption_id']
            if mp.get(caption_id) is None:
                data[j]['captions'][i]['edited'] = tt['id']
            else:
                data[j]['captions'][i]['edited'] = mp[caption_id][0]
    with open(dataset_folder + '/caption.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    path = './dataset'
    flickr_folder = path + '/flickr10k'
    coco_folder = path + '/mscoco'

    # convert mongo to caption dict
    caption_flickr, caption_coco = convert_mongo(path=path + '/caption_bc.json')

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
