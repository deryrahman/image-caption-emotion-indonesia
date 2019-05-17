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
from model import EncoderCNN, DecoderRNN
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
    mode = args.mode

    image_dir = args.image_dir
    caption_path = args.caption_path
    val_caption_path = args.val_caption_path
    val_emotion_path = args.val_emotion_path
    caption_batch_size = args.caption_batch_size

    emotion_path = args.emotion_path
    language_batch_size = args.language_batch_size

    embed_size = args.embed_size
    hidden_size = args.hidden_size
    dropout = args.dropout

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
    val_data_loader = get_loader(image_dir,
                                 val_caption_path,
                                 vocab,
                                 transform,
                                 caption_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    if mode != 'factual':
        emotion_data_loader = get_loader(image_dir,
                                         emotion_path,
                                         vocab,
                                         transform,
                                         language_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)
        val_emotion_data_loader = get_loader(image_dir,
                                             val_emotion_path,
                                             vocab,
                                             transform,
                                             language_batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size,
                         hidden_size,
                         len(vocab),
                         1,
                         dropout=dropout).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters()) + list(
        encoder.linear.parameters()) + list(encoder.bn.parameters())
    lang_params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr_caption)
    lang_optimizer = torch.optim.Adam(lang_params, lr=lr_language)

    # Train the models
    for epoch in range(num_epochs):
        # train factual
        res = train(encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    criterion=criterion,
                    data_loader=data_loader,
                    log_step=log_step)

        val_res = val(encoder=encoder,
                      decoder=decoder,
                      vocab=vocab,
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

        # Save the model checkpoints
        torch.save(
            decoder.state_dict(),
            os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
        torch.save(
            encoder.state_dict(),
            os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))

    if mode != 'factual':
        for epoch in range(num_epochs):
            # train style
            res = train(encoder=encoder,
                        decoder=decoder,
                        optimizer=lang_optimizer,
                        criterion=criterion,
                        data_loader=emotion_data_loader,
                        log_step=log_step_emotion)
            val_res = val(encoder=encoder,
                          decoder=decoder,
                          vocab=vocab,
                          criterion=criterion,
                          data_loader=val_emotion_data_loader)
            batch_time, loss = res
            val_batch_time, top5, loss_val = val_res
            batch_time += val_batch_time
            print(
                """Epoch [{}/{}], [{}], Batch Time: {:.3f}, Top-5 Acc: {:.3f}"""
                .format(epoch, num_epochs, mode[:3].upper(), batch_time, top5))
            print("""\tTrain Loss: {:.4f} | Train Perplexity: {:5.4f}""".format(
                loss, np.exp(loss)))
            print("""\tVal   Loss: {:.4f} | Val   Perplexity: {:5.4f}""".format(
                loss_val, np.exp(loss_val)))

        # Save the model checkpoints
        torch.save(
            decoder.state_dict(),
            os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
        torch.save(
            encoder.state_dict(),
            os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))


def val(encoder, decoder, vocab, criterion, data_loader):
    decoder.eval()
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
        outputs = decoder(captions, lengths, features, teacher_forcing_ratio=0)
        loss = criterion(outputs, targets)

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        top5 = accuracy(outputs, targets, 5)
        top5accs.update(top5, sum(lengths))
        batch_time.update(time.time() - start)

    feature = features[0].unsqueeze(0)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break

    print(sampled_caption)

    return batch_time.val, top5accs.avg, losses.avg


def train(encoder, decoder, optimizer, criterion, data_loader, log_step):
    decoder.train()
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
        outputs = decoder(captions, lengths, features)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related
    parser.add_argument('--model_path',
                        type=str,
                        default='models/',
                        help='path for saving trained models')
    parser.add_argument('--mode',
                        type=str,
                        default='factual',
                        help='mode training, factual, happy, sad, angry')
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
    parser.add_argument('--emotion_path',
                        type=str,
                        default='data/flickr8k_id/happy/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_emotion_path',
                        type=str,
                        default='data/flickr8k_id/happy/val.txt',
                        help='path for train txt file')

    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--log_step_emotion', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.22)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--caption_batch_size', type=int, default=64)
    parser.add_argument('--language_batch_size', type=int, default=96)
    parser.add_argument('--lr_caption', type=float, default=0.0002)
    parser.add_argument('--lr_language', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
