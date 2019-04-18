from preparation import split_dataset, save_dataset
from tokenizer import mark_captions, flatten, TokenizerWrap
import json

path = './dataset'
flickr_folder = path + '/flickr10k'

with open(flickr_folder + '/captions.json', 'r') as f:
    caption_flickr = json.load(f)

all_filenames = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
all_captions = {'factual': [], 'happy': [], 'sad': [], 'angry': []}
for mode in ['happy', 'sad', 'angry']:
    for data in caption_flickr:
        if data['emotions'].get(mode):
            all_filenames[mode].append(data['filename'])
            all_captions[mode].append([data['emotions'][mode]])
all_filenames['factual'] = [data['filename'] for data in caption_flickr]
all_captions['factual'] = [
    [caption['id'] for caption in data['captions']] for data in caption_flickr
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
        all_filenames[mode], all_captions[mode], flickr_folder + '/' + mode,
        tokenizer, 10, 10)
    save_dataset(all_filenames[mode], all_captions[mode],
                 flickr_folder + '/' + mode, train_indexes, val_indexes,
                 test_indexes)
