import json
import os


def convert_mongo(path):
    """Convert mongo dump from imagecaption.geekstudio.id into list of captions

    Arguments:
        path {str} -- string of path for mongo dump

    Returns:
        list pair -- caption_flickr, caption_coco. Each list handle list
        of captions with format:
        {
            'image_id': ...,
            'filename': ...,
            'captions': [
                ...,
                ...
            ],
            'emotions': {
                'happy': ...,
                'sad': ...,
                'angry': ...
            }
        }
    """

    caption_flickr = []
    caption_coco = []
    with open(path) as f:
        content = f.readlines()
        for d in content:
            tmp = json.loads(d.rstrip())
            data = {
                'image_id': tmp['image_id'],
                'filename': tmp['file_name'],
                'captions': tmp['captions'],
                'emotions': {}
            }
            if tmp['image_id'].split('-')[0] == 'coco':
                caption_coco.append(data)
            else:
                caption_flickr.append(data)
    return caption_flickr, caption_coco


def split_dataset(filenames,
                  captions,
                  path_folder,
                  tokenizer,
                  val_percentage=10,
                  test_percentage=10):
    """Split given list of filenames and lisf of captions into train, val, and test set.
    We need to have all word in training data. So before it, we need to do heuristic function,
    to sort the captions based on word frequency count. This can be done with
    _sort_captions_by_largest_word_freq function. Then, iterate each caption, if words in that caption
    have frequency larger than 1, then it's safe to take the caption as validation or test set.
    This can be done with _split_train_val_test function

    Arguments:
        filenames {list} -- list of filenames. eg ['123.jpg', '124.jpg', '448.jpg']
        captions {list of list} -- list of captions. eg [
            ['1 ref a', '1 ref b', '1 ref c'], ['2 ref a', '2 ref b', '2 ref c']
            ]
        path_folder {str} -- folder path to save the result
        tokenizer {tokenizer} -- tokenizer object

    Keyword Arguments:
        val_percentage {int} -- validation percentage (default: {10})
        test_percentage {int} -- test percentage (default: {10})

    Returns:
        list, list, list -- list of train indexes, val indexes, and test indexes
    """

    val_num = len(captions) * val_percentage // 100
    test_num = len(captions) * test_percentage // 100
    pair_all_freq_ids = _sort_captions_by_largest_word_freq(captions, tokenizer)
    train_ids, val_ids, test_ids = _split_train_val_test(
        pair_all_freq_ids, captions, tokenizer, val_num, test_num)

    train_indexes = [filenames[i].split('.')[0] for i in train_ids]
    val_indexes = [filenames[i].split('.')[0] for i in val_ids]
    test_indexes = [filenames[i].split('.')[0] for i in test_ids]

    return train_indexes, val_indexes, test_indexes


def save_dataset(filenames, captions, path_folder, train_indexes, val_indexes,
                 test_indexes):
    """Save dataset that has been splitted into train, val, and test

    Arguments:
        path_folder {str} -- string of path dataset folder. eg ./dataset/flickr10k/factual
        filenames {list} -- list of filename
        captions {list of list} -- list of captions
        train_indexes {list} -- list of train indexes
        val_indexes {list} -- list of validation indexes
        test_indexes {test} -- list of test indexes
    """

    if not os.path.isdir(path_folder):
        os.mkdir(path_folder)

    with open(path_folder + '/filenames.json', 'w') as f:
        json.dump(filenames, f)
    with open(path_folder + '/captions.json', 'w') as f:
        json.dump(captions, f)
    with open(path_folder + '/train.txt', 'w') as f:
        f.write('\n'.join(train_indexes))
    with open(path_folder + '/val.txt', 'w') as f:
        f.write('\n'.join(val_indexes))
    with open(path_folder + '/test.txt', 'w') as f:
        f.write('\n'.join(test_indexes))


def _sort_captions_by_largest_word_freq(captions, tokenizer):
    """Sort captions based on word count frequency within captions.
    It uses for splitting heuristic, because we need to have all word in training dataset.
    This function is used to sort the caption from largest word count frequency.

    So with this frequency heuristic, we can split validation and test

    Arguments:
        captions {list of list} -- list of captions
        tokenizer {tokenizer} -- tokenizer object

    Returns:
        list of tuple -- pair of frequency and id (freq,id)
    """

    pair_freq_ids = []

    for i in range(len(captions)):
        caps = captions[i]
        total = 0
        length = 0
        for cap in tokenizer.texts_to_sequences(caps):
            for word_i in cap:
                length += 1
                word = tokenizer.index_to_word[word_i]
                total += tokenizer.word_counts[word]
        pair_freq_ids.append((total / length, i))
    pair_freq_ids = sorted(pair_freq_ids, key=lambda x: x[0], reverse=True)
    return pair_freq_ids


def _split_train_val_test(pair_freq_ids, captions, tokenizer, val_num,
                          test_num):
    """Split train, validation, and test set. It will calculate the words frequency
    in each caption. If the words have frequency larger than 1, then it's safe to take it
    as validation or test set

    Arguments:
        pair_freq_ids {list of tuple} -- list of tuple frequency and ids. (freq, id)
        captions {list of list} -- list of captions
        tokenizer {tokenizer} -- tokenizer object
        val_num {int} -- how many validation data that need to take
        test_num {int} -- how many test data that need to take

    Returns:
        list, list, list -- list of ids for train, validation, and test set
    """

    word_counts = tokenizer.word_counts
    train_ids = []
    val_ids = []
    test_ids = []

    for total, index in pair_freq_ids:
        caps = captions[index]
        is_safe = True
        for cap in tokenizer.texts_to_sequences(caps):
            if not is_safe:
                break
            for word_i in cap:
                if not is_safe:
                    break
                word = tokenizer.index_to_word[word_i]
                cnt = word_counts[word]
                word_counts[word] -= 1
                is_safe = cnt > 1
        if is_safe and len(val_ids) < val_num:
            val_ids.append(index)
        elif is_safe and len(test_ids) < test_num:
            test_ids.append(index)
        else:
            train_ids.append(index)

    return train_ids, val_ids, test_ids


def load_caption(path_folder):
    """Load caption from folder. The folder must has the structure like this:
    /filenames.json -- contain list of filenames
    /captions.json -- contain list of list of caption
    /train.txt -- contain all train indexes
    /val.txt -- contain all validation indexes
    /test.txt -- contain all test indexes

    Arguments:
        path_folder {str} -- string of path folder

    Returns:
        tuple, tuple, tuple -- three tuples of train, val, and test.
        Each tuple has filenames and captions (filename, caption)
    """

    with open(path_folder + '/filenames.json', 'r') as f:
        filenames = json.load(f)
    with open(path_folder + '/captions.json', 'r') as f:
        captions = json.load(f)
    with open(path_folder + '/train.txt', 'r') as f:
        train_indexes = [d.rstrip() for d in f.readlines()]
    with open(path_folder + '/val.txt', 'r') as f:
        val_indexes = [d.rstrip() for d in f.readlines()]
    with open(path_folder + '/test.txt', 'r') as f:
        test_indexes = [d.rstrip() for d in f.readlines()]

    filenames_train, captions_train = _filter_data_by_indexes(
        filenames=filenames, captions=captions, indexes=train_indexes)
    filenames_val, captions_val = _filter_data_by_indexes(
        filenames=filenames, captions=captions, indexes=val_indexes)
    filenames_test, captions_test = _filter_data_by_indexes(
        filenames=filenames, captions=captions, indexes=test_indexes)

    train = (filenames_train, captions_train)
    val = (filenames_val, captions_val)
    test = (filenames_test, captions_test)
    return train, val, test


def _filter_data_by_indexes(filenames, captions, indexes):
    """This function is used to filter the files and captions based on the indexes

    Arguments:
        filenames {list} -- list of filenames
        captions {list of list} -- list of list of captions
        indexes {list} -- list of indexes

    Returns:
        list, list -- filtered filenames and filtered captions
    """

    filtered_filenames = ()
    filtered_captions = ()
    for filename, caption in zip(filenames, captions):
        index = filename.split('.')[0]
        if index in indexes:
            filtered_filenames += (filename,)
            filtered_captions += ([cap for cap in caption],)
    return filtered_filenames, filtered_captions


def invoke_emotion_to_dataset(mongo_dump_path, dataset_folder, dataset,
                              emotion):
    """This function is used to invoke the emotion into current dataset

    Arguments:
        mongo_dump_path {str} -- string of path of the original captions.json
        dataset_folder {str} -- string of dataset folder
        dataset {str} -- either flickr or coco
        emotion {str} -- which emotion that you want to invoke

    Raises:
        ValueError -- raises when dataset neither flickr nor coco

    Returns:
        list -- list of dict captions
    """

    if dataset != 'flickr' and dataset != 'coco':
        raise ValueError('dataset only flickr or coco')

    with open(dataset_folder + '/captions.json', 'r') as f:
        captions = json.load(f)

    image_id_to_idx = _generate_image_id_to_ids(dataset_folder)

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
                captions[image_id_to_idx[user_caption['image_id']]]['emotions'][
                    emotion] = user_caption['captionEmotion'][emotion]

    with open(dataset_folder + '/captions.json', 'w') as f:
        json.dump(captions, f)

    return captions


def _generate_image_id_to_ids(dataset_folder):
    """Function to help generate map of image_id and idx map[image_id] = idx

    Arguments:
        dataset_folder {str} -- string of dataset folder

    Returns:
        map -- map[image_id] = idx
    """

    image_id_to_idx = {}
    with open(dataset_folder + '/captions.json', 'r') as f:
        contents = json.load(f)

    for i, data in enumerate(contents):
        image_id_to_idx[data['image_id']] = i
    return image_id_to_idx


def invoke_edited_to_dataset(mongo_dump_path, dataset_folder):
    """This function is used to invoke edited caption into current dataset

    Arguments:
        mongo_dump_path {str} -- string of mongo dump path
        dataset_folder {str} -- string of dataset folder
    """

    mp = {}
    with open(mongo_dump_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            user = json.loads(line)
            for cap in user['captions']:
                if len(cap['captionEdit']) == 0:
                    continue
                for tmp in cap['captionEdit']:
                    caption_id = tmp['caption_id']
                    if mp.get(caption_id) is None:
                        mp[caption_id] = []
                    mp[caption_id].append(tmp['text'])

    with open(dataset_folder + '/captions.json', 'r') as f:
        contents = json.load(f)

    for i in range(len(contents)):
        for j, cap in enumerate(contents[i]['captions']):
            # for flickr30k. only flickr10k that has caption_id
            if cap.get('caption_id') is None:
                continue
            caption_id = cap['caption_id']
            if mp.get(caption_id) is None:
                contents[i]['captions'][j]['edited'] = cap['id']
            else:
                contents[i]['captions'][j]['edited'] = mp[caption_id][0]

    with open(dataset_folder + '/captions.json', 'w') as f:
        json.dump(contents, f)
