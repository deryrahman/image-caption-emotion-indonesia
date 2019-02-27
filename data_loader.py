from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
import glob
import numpy as np
import json


class ImageLoader():
    """Helper class for load image"""

    def load(self, filename, target_size=(224, 224), extension='jpg'):

        if extension != 'jpg':
            raise Exception('{} must be jpg'.format(extension))

        img = image.load_img(filename, target_size=target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        return img

    def load_from_folder(self,
                         path_folder,
                         target_size=(224, 224),
                         extension='jpg'):

        filenames = glob.glob('{}/*.{}'.format(path_folder, extension))
        images_bytes = {}

        for filename in filenames:
            name = filename.split('/')[-1].split('.')[-2]
            images_bytes[name] = self.load(
                filename, target_size=target_size, extension=extension)

        return images_bytes


class CaptionLoader():
    """Helper class for load captions"""

    def load(self,
             filename,
             preprocess=lambda text: text,
             caption_modes=['id', 'en']):
        mapping_caption = {mode: {} for mode in caption_modes}

        with open(filename, 'r') as f:
            data = json.load(f)

        for d in data:
            k = d['image_id']
            for mode in caption_modes:
                mapping_caption[mode][k] = []
            for captions in d['captions']:
                for mode in caption_modes:
                    captions[mode] = preprocess(captions[mode])
                    mapping_caption[mode][k].append(captions[mode])

        return mapping_caption
