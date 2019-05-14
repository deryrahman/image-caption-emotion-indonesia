import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
from build_vocab import Vocabulary
import re


class FlickrDataset(data.Dataset):
    __cache = {}

    def __init__(self, img_dir, caption_file, vocab, transform=None):
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.vocab = vocab
        self.transform = transform

    def _get_imgname_and_caption(self, caption_file):
        with open(caption_file, 'r') as f:
            res = f.readlines()

        imgname_caption_list = []
        r = re.compile(r'#\d*')
        for line in res:
            img_and_cap = r.split(line)
            img_and_cap = [x.strip() for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __getitem__(self, index):

        img_name = self.imgname_caption_list[index][0]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[index][1]

        image = FlickrDataset.__cache.get(img_name)
        if image is None:
            image = Image.open(img_name).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            FlickrDataset.__cache[img_name] = image

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.imgname_caption_list)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(img_dir, caption_file, vocab, transform, batch_size, shuffle,
               num_workers):
    """Returns torch.utils.data.DataLoader for custom flickr dataset."""
    # Flickr caption dataset
    flickr = FlickrDataset(img_dir=img_dir,
                           caption_file=caption_file,
                           vocab=vocab,
                           transform=transform)

    # Data loader for Flickr dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=flickr,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
