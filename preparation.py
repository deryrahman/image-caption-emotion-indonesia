from data_loader import ImageLoader, CaptionLoader
from model import EncoderResNet152
import json
import pickle
import string
import numpy as np


def load_img_features(data_filename, model, img_dir):
    with open(data_filename, 'r') as f:
        data = json.load(f)

    mapping_img = {}
    image_loader = ImageLoader()

    for i, d in enumerate(data):
        try:
            k = d['image_id']
            img_path = '{}/{}'.format(img_dir, d['file_name'])
            x = image_loader.load(img_path)
            mapping_img[k] = model.predict(x)
        except Exception as e:
            print(e)
        if i % 10 == 0:
            print('{}/{}'.format(i, len(data)))

    return mapping_img


def load_captions(data_filename):
    caption_loader = CaptionLoader()
    mapping_captions = caption_loader.load(
        data_filename, preprocess=preprocess_text)
    return mapping_captions


def save(data, filename):
    pickle.dump(data, open(filename, 'wb'))


def load(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_text(text):
    for p in string.punctuation:
        text = text.replace(p, ' ')
    return '<start> ' + ' '.join(t.lower() for t in text.split()) + ' <end>'


if __name__ == '__main__':
    flickr_path = './dataset/flickr10k/flickr_10k.json'
    image_path = './dataset/flickr10k/img/'
    features_path = './pretrained/features.pkl'
    caption_data_path = './dataset/flickr10k/caption.pkl'
    train_data_path = './dataset/flickr10k/train.pkl'
    validation_data_path = './dataset/flickr10k/validation.pkl'

    model = EncoderResNet152().get_model()
    features = load_img_features(flickr_path, model, image_path)
    save(features, features_path)

    captions = load_captions(flickr_path)

    for mode in ['id', 'en']:
        # split data training and validation
        indexes = np.array([k for k, v in captions[mode].items()])
        np.random.seed(0)
        np.random.shuffle(indexes)
        train_indexes = indexes[-9000:,]
        validation_indexes = indexes[:1000,]

        train_data = {}
        validation_data = {}
        for i in train_indexes:
            train_data[i] = captions[mode][i]
        for i in validation_indexes:
            validation_data[i] = captions[mode][i]

        save(captions[mode], caption_data_path + '.' + mode)
        save(train_data, train_data_path + '.' + mode)
        save(validation_data, validation_data_path + '.' + mode)
