import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
import time
from data_loader import get_loader, get_style_loader
from build_vocab import Vocabulary
from model import EncoderCNN, Seq2Seq
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import AverageMeter, accuracy
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
    factual_caption_path = args.factual_caption_path
    val_image_dir = args.val_image_dir
    val_caption_path = args.val_caption_path
    caption_batch_size = args.caption_batch_size

    happy_path = args.happy_path
    sad_path = args.sad_path
    angry_path = args.angry_path
    language_batch_size = args.language_batch_size

    embed_size = args.embed_size
    hidden_size = args.hidden_size

    lr_caption = args.lr_caption
    lr_language = args.lr_language
    num_epochs = args.num_epochs
    log_step = args.log_step
    log_step_emotion = args.log_step_emotion

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
    val_data_loader = get_loader(val_image_dir,
                                 val_caption_path,
                                 vocab,
                                 transform,
                                 caption_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    happy_data_loader = get_style_loader(image_dir,
                                         factual_caption_path,
                                         happy_path,
                                         vocab,
                                         transform,
                                         language_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)
    sad_data_loader = get_style_loader(image_dir,
                                       factual_caption_path,
                                       sad_path,
                                       vocab,
                                       transform,
                                       language_batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
    angry_data_loader = get_style_loader(image_dir,
                                         factual_caption_path,
                                         angry_path,
                                         vocab,
                                         transform,
                                         language_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    seq2seq = Seq2Seq(embed_size, hidden_size, len(vocab), 1).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    params = list(seq2seq.encoder.parameters()) + list(
        encoder.linear.parameters()) + list(encoder.bn.parameters())
    happy_params = list(seq2seq.decoder_happy.parameters())
    sad_params = list(seq2seq.decoder_sad.parameters())
    angry_params = list(seq2seq.decoder_angry.parameters())

    optimizer = torch.optim.Adam(params, lr=lr_caption)
    happy_optimizer = torch.optim.Adam(happy_params, lr=lr_language)
    sad_optimizer = torch.optim.Adam(sad_params, lr=lr_language)
    angry_optimizer = torch.optim.Adam(angry_params, lr=lr_language)

    # Train the models
    data_loaders = [happy_data_loader, sad_data_loader, angry_data_loader]
    tags = ['happy', 'sad', 'angry']
    optimizers = [happy_optimizer, sad_optimizer, angry_optimizer]

    # Train the models
    for epoch in range(num_epochs):
        # train factual
        res = train_factual(encoder=encoder,
                            seq2seq=seq2seq,
                            optimizer=optimizer,
                            criterion=criterion,
                            data_loader=data_loader,
                            log_step=log_step)

        val_res = val_factual(encoder=encoder,
                              seq2seq=seq2seq,
                              criterion=criterion,
                              data_loader=val_data_loader)
        batch_time, loss = res
        val_batch_time, top5, loss_val = val_res
        batch_time += val_batch_time
        print("""Epoch [{}/{}], [FAC], Batch Time: {:.3f}, Top-5 Acc: {:.3f}""".
              format(epoch, num_epochs, batch_time, top5))
        print("""\tTrain Loss: {:.4f} | Train Perplexity: {:5.4f}""".format(
            loss, np.exp(loss)))
        print("""\tVal   Loss: {:.4f} | Val   Perplexity: {:5.4f}""".format(
            loss_val, np.exp(loss_val)))

        # train style

        res = train_emotion(encoder=encoder,
                            seq2seq=seq2seq,
                            optimizers=optimizers,
                            criterion=criterion,
                            data_loaders=data_loaders,
                            tags=tags,
                            log_step=log_step_emotion)
        batch_time, losses = res
        print("""Batch Time: {:.3f}""".format(batch_time))
        for i in range(len(tags)):
            print(
                """Epoch [{}/{}], [{}], Train Loss: {:.4f}, Train Perplexity: {:5.4f}"""
                .format(epoch, num_epochs, tags[i][:3].upper(), losses[i],
                        np.exp(losses[i])))

        # Save the model checkpoints
        torch.save(
            seq2seq.state_dict(),
            os.path.join(args.model_path, 'seq2seq-{}.ckpt'.format(epoch + 1)))
        torch.save(
            encoder.state_dict(),
            os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))


def val_factual(encoder, seq2seq, criterion, data_loader):
    seq2seq.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(input=captions,
                                       lengths=lengths,
                                       batch_first=True)[0]
        # Forward, backward and optimize
        features = encoder(images)
        outputs = seq2seq(features, (captions, lengths))
        loss = criterion(outputs, targets)

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        top5 = accuracy(outputs, targets, 5)
        top5accs.update(top5, sum(lengths))
        batch_time.update(time.time() - start)

    return batch_time.val, top5accs.avg, losses.avg


def train_factual(encoder, seq2seq, optimizer, criterion, data_loader,
                  log_step):
    seq2seq.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(input=captions,
                                       lengths=lengths,
                                       batch_first=True)[0]
        # Forward, backward and optimize
        features = encoder(images)
        outputs = seq2seq(features, (captions, lengths))
        loss = criterion(outputs, targets)
        seq2seq.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_step == 0:
            print("""Step [{}/{}], [FAC], Loss: {:.4f}""".format(
                i, len(data_loader), loss.item()))

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        batch_time.update(time.time() - start)

    return batch_time.val, losses.avg


def train_emotion(encoder, seq2seq, optimizers, criterion, data_loaders, tags,
                  log_step):
    seq2seq.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(tags))]
    start = time.time()

    for j in random.sample([i for i in range(len(tags))], len(tags)):
        for i, (images, src, dst) in enumerate(data_loaders[j]):
            # Set mini-batch dataset
            images = images.to(device)
            captions_src, length_src = src
            captions_dst, length_dst = src
            captions_src = captions_src.to(device)
            captions_dst = captions_dst.to(device)
            length_dst = [l - 1 for l in length_dst]
            targets = pack_padded_sequence(input=captions_dst[:, 1:],
                                           lengths=length_dst,
                                           batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            outputs = seq2seq(features=features,
                              src=(captions_src, length_src),
                              dst=(captions_dst[:, :-1], length_dst),
                              mode=tags[j])
            loss = criterion(outputs, targets)
            seq2seq.zero_grad()
            # encoder.zero_grad()
            loss.backward()
            optimizers[j].step()

            if i % log_step == 0:
                print("""Step [{}/{}], [{}], Loss: {:.4f}""".format(
                    i, len(data_loaders[j]), tags[j][:3].upper(), loss.item()))
            # Keep track of metrics
            losses[j].update(loss.item(), sum(length_dst))
            batch_time.update(time.time() - start)

    return batch_time.val, [loss.avg for loss in losses]


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
    parser.add_argument('--factual_caption_path',
                        type=str,
                        default='data/flickr30k_id/all.txt',
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
    parser.add_argument('--happy_path',
                        type=str,
                        default='data/flickr8k_id/happy/train_supervised.txt',
                        help='path for train txt file')
    parser.add_argument('--sad_path',
                        type=str,
                        default='data/flickr8k_id/sad/train_supervised.txt',
                        help='path for train txt file')
    parser.add_argument('--angry_path',
                        type=str,
                        default='data/flickr8k_id/angry/train_supervised.txt',
                        help='path for train txt file')

    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--log_step_emotion', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--caption_batch_size', type=int, default=64)
    parser.add_argument('--language_batch_size', type=int, default=96)
    parser.add_argument('--lr_caption', type=float, default=0.0002)
    parser.add_argument('--lr_language', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
