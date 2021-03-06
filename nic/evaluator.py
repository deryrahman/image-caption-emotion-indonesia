import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
import time
from data_loader import get_loader
from build_vocab import Vocabulary
from model_att import EncoderCNN, DecoderRNNAtt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchvision import transforms
from utils import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint
from nltk.translate.bleu_score import corpus_bleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    checkpoint_path = args.checkpoint_path
    # mode = args.mode
    image_dir = args.image_dir
    test_path = args.test_path
    vocab_path = args.vocab_path
    language_batch_size = 96
    num_workers = 0

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    torch.nn.Module.dump_patches = True
    checkpoint = torch.load(checkpoint_path, map_location=device)
    last_epoch = checkpoint['epoch']
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    print('last_epoch', last_epoch)

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    test_data_loader = get_loader(image_dir,
                                  test_path,
                                  vocab,
                                  transform,
                                  language_batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    decoder.eval()
    encoder.eval()

    # references (true captions) for calculating BLEU-4 score
    references = list()
    # hypotheses (predictions)
    hypotheses = list()
    for i, (images, captions, lengths,
            all_captions) in enumerate(test_data_loader):
        # Set mini-batch dataset
        images = images.to(device)

        # Forward, backward and optimize
        with torch.no_grad():
            features = encoder(images)

        start = vocab.word2idx['<start>']
        end = vocab.word2idx['<end>']
        for feature, caps in zip(features, all_captions):
            feature = feature.unsqueeze(0)

            sampled_ids = decoder.sample(feature,
                                         start_token=start,
                                         end_token=end)
            sampled_ids = sampled_ids[0].cpu().numpy()
            caps = [c.long().tolist() for c in caps]
            references.append(caps)

            sampled_caption = []
            for word_id in caps[0]:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            print('ref', ' '.join(sampled_caption))

            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            print('pred', ' '.join(sampled_caption))

            hypotheses.append(sampled_ids)

    assert len(references) == len(hypotheses)

    bleu_1 = corpus_bleu(list_of_references=references,
                         hypotheses=hypotheses,
                         weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(list_of_references=references,
                         hypotheses=hypotheses,
                         weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(list_of_references=references,
                         hypotheses=hypotheses,
                         weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(list_of_references=references,
                         hypotheses=hypotheses,
                         weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1', bleu_1)
    print('BLEU-2', bleu_2)
    print('BLEU-3', bleu_3)
    print('BLEU-4', bleu_4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related
    parser.add_argument('--checkpoint_path', type=str)
    # parser.add_argument('--mode', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--vocab_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args)

    # checkpoint_path = './models/stylenet_multitask_att_4_complete/ANG_BEST_checkpoint_stylenet_multitask_att_4.pth.tar'
    # mode = 'angry'
    # image_dir = '/Users/dery/Documents/final_project/dataset/flickr30k/flickr30k_images'
    # test_path = '/Users/dery/Documents/final_project/13515097-stylenet/data/flickr8k_id/angry/test.txt'
    # vocab_path = '/Users/dery/Documents/final_project/13515097-stylenet/data/flickr8k_id/vocab_051819_4.pkl'