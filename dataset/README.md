# Dataset

Contain dataset of partial MSCOCO and Flickr30k

### Path
```
    - flickr10k -- contain flickr dataset
    - mscoco -- contain mscoco dataset
    - dump -- contain mongo dump from imagecaption.geekstudio.id
    - cache -- contain helper, such image id to index
```

### Data
```
path format :
    - /img/ -- contain images
    - /cache/ -- contain cache
    - /caption.json -- contain caption
    - /train.txt -- contain list of filename train
    - /val.txt -- contain list of filename validation

format caption.json :
    - {
        'filename': 000123.jpg
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