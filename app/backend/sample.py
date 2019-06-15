import torch
import pickle
from torchvision import transforms
from model import load_model
from model_att import load_model_att
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def load_image(image_path, transform=None):
    image = Image.open(image_path)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def load_vocab(vocab_path):
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    return vocab


def get_sample(checkpoint_path, vocab_path, mode, with_att, image):

    if with_att:
        encoder, decoder = load_model_att(checkpoint_path)
    else:
        encoder, decoder = load_model(checkpoint_path)

    vocab = load_vocab(vocab_path)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Prepare an image
    image = load_image(image, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    with torch.no_grad():
        feature = encoder(image_tensor)
    feature = feature.unsqueeze(0)
    start = vocab.word2idx['<start>']
    end = vocab.word2idx['<end>']
    sampled_ids = decoder.sample(
        feature,
        start_token=start,
        end_token=end,
        #  k=3,
        mode=mode)
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sampled_caption = [
        w for w in sampled_caption if w != '<start>' and w != '<end>'
    ]
    if sampled_caption[-1] == '<unk>':
        sampled_caption = sampled_caption[:-1]

    return ' '.join(w for w in sampled_caption)
