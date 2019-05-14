import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# from validate import validate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(0)


def main(args):
    model_path = args.model_path
    crop_size = args.crop_size
    vocab_path = args.vocab_path
    num_workers = args.num_workers

    image_dir = args.image_dir
    caption_path = args.caption_path
    caption_batch_size = args.caption_batch_size

    embed_size = args.embed_size
    hidden_size = args.hidden_size

    lr_caption = args.lr_caption
    num_epochs = args.num_epochs
    log_step = args.log_step

    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(image_dir,
                             caption_path,
                             vocab,
                             transform,
                             caption_batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), 1).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters()) + list(
        encoder.linear.parameters()) + list(encoder.bn.parameters())

    optimizer = torch.optim.Adam(params, lr=lr_caption)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        decoder.train()
        encoder.train()

        for i, (images, captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(input=captions,
                                           lengths=lengths,
                                           batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(captions, lengths, features)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                print(
                    'Epoch [{}/{}], [FAC], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss.item(),
                            np.exp(loss.item())))

        # Save the model checkpoints
        torch.save(
            decoder.state_dict(),
            os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
        torch.save(
            encoder.state_dict(),
            os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related
    parser.add_argument('--model_path',
                        type=str,
                        default='models/',
                        help='path for saving trained models')
    parser.add_argument('--vocab_path',
                        type=str,
                        default='data/flickr8k_id/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir',
                        type=str,
                        default='/home/m13515097/final_project/'
                        '13515097-stylenet/dataset/flickr30k/img',
                        help='directory for train images')
    parser.add_argument('--caption_path',
                        type=str,
                        default='data/flickr8k_id/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_image_dir',
                        type=str,
                        default='/home/m13515097/final_project/'
                        '13515097-stylenet/dataset/flickr30k/img',
                        help='directory for val images')
    parser.add_argument('--val_caption_path',
                        type=str,
                        default='data/flickr8k_id/val.txt',
                        help='path for val txt file')

    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--crop_size', type=int, default=224)

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--factored_size', type=int, default=512)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--caption_batch_size', type=int, default=64)
    parser.add_argument('--language_batch_size', type=int, default=96)
    parser.add_argument('--lr_caption', type=float, default=0.0002)
    parser.add_argument('--lr_language', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
