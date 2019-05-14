import torch
# import matplotlib.pyplot as plt
# import numpy as np
import argparse
import pickle
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # load dataset
    cap_l = []
    with open(args.test_caption_path, 'r') as f:
        cap_l = f.readlines()

    # Build models
    encoder = EncoderCNN(args.embed_size).eval(
    )  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab),
                         args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    text = ''
    text_ref = ''
    for li in cap_l:
        # Prepare an image
        fn = li.split('#')[0]
        ref = li.split('\t')[-1]
        text_ref += ref
        image = load_image(args.test_image_dir + '/' + fn, transform)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        # (1, max_seq_length) -> (max_seq_length)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        sentence = sentence.replace('<start>', '')
        sentence = sentence.replace('<end>', '')
        sentence = sentence.strip()

        # Print out the image and the generated caption
        print(sentence)
        text += sentence + '\n'
        # image = Image.open(args.image)
        # plt.imshow(np.asarray(image))
    with open(args.result_dir + '/hyp.txt', 'w') as f:
        f.write(text)
    with open(args.result_dir + '/ref.txt', 'w') as f:
        f.write(text_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_dir',
                        type=str,
                        default=('/Users/dery/Documents/final_project/'
                                 '13515097-stylenet/dataset/flickr30k/img'),
                        help='directory for test images')
    parser.add_argument('--test_caption_path',
                        type=str,
                        default='data/flickr8k_id/happy/train_supervised.txt',
                        help='path for test txt file')
    parser.add_argument('--result_dir',
                        type=str,
                        default='result/flickr8k_id/happy/',
                        help='path for test txt file')
    parser.add_argument('--encoder_path',
                        type=str,
                        default='models/encoder-2-1000.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path',
                        type=str,
                        default='models/decoder-2-1000.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path',
                        type=str,
                        default='data/vocab.pkl',
                        help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size',
                        type=int,
                        default=300,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--factored_size',
                        type=int,
                        default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
