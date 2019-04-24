# Dataset

Contain dataset of partial MSCOCO and Flickr30k

### Path
```
    - flickr10k -- contain flickr dataset
    - mscoco -- contain mscoco dataset
    - dump -- contain mongo dump from imagecaption.geekstudio.id
```

### Data
```
path format :
    - /img/ -- contain images
    - /cache/ -- contain cache (tokenizer dump, transfer_values)
    - /captions.json -- contain all captions data
    - /factual
        - /captions.json -- only contain list of list captions
        - /filenames.json -- list of filenames
        - train.txt -- indexes for train
        - val.txt -- indexes for validation
        - test.txt -- indexes for test

format captions.json (all captions data version):
    - {
        'image_id': 331-flickr,
        'filename': 000123.jpg,
        'captions': [
            {'id': 'Bahasa Indonesia',
             'en' : 'English'},
            ...
        ],
        'emotions': {
            'happy': 'Emosi happy',
            'sad': 'Emosi sad',
            'angry': 'Emosi angry'
        }
      }
```