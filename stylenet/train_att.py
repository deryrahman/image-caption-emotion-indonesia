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
from model_att import EncoderCNN, DecoderFactoredLSTMAtt
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
    val_caption_path = args.val_caption_path
    val_happy_path = args.val_happy_path
    val_sad_path = args.val_sad_path
    val_angry_path = args.val_angry_path
    caption_batch_size = args.caption_batch_size

    happy_path = args.happy_path
    sad_path = args.sad_path
    angry_path = args.angry_path
    language_batch_size = args.language_batch_size

    embed_size = args.embed_size
    hidden_size = args.hidden_size
    factored_size = args.factored_size
    attention_size = args.attention_size
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
    happy_data_loader = get_loader(image_dir,
                                   happy_path,
                                   vocab,
                                   transform,
                                   language_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
    val_happy_data_loader = get_loader(image_dir,
                                       val_happy_path,
                                       vocab,
                                       transform,
                                       language_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)
    sad_data_loader = get_loader(image_dir,
                                 sad_path,
                                 vocab,
                                 transform,
                                 language_batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
    val_sad_data_loader = get_loader(image_dir,
                                     val_sad_path,
                                     vocab,
                                     transform,
                                     language_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
    angry_data_loader = get_loader(image_dir,
                                   angry_path,
                                   vocab,
                                   transform,
                                   language_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
    val_angry_data_loader = get_loader(image_dir,
                                       val_angry_path,
                                       vocab,
                                       transform,
                                       language_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN().to(device)
    decoder = DecoderFactoredLSTMAtt(attention_size,
                                     embed_size,
                                     hidden_size,
                                     factored_size,
                                     len(vocab),
                                     1,
                                     dropout=dropout).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(
        encoder.adaptive_pool.parameters())
    lang_params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr_caption)
    lang_optimizer = torch.optim.Adam(lang_params, lr=lr_language)

    # Train the models
    data_loaders = [happy_data_loader, sad_data_loader, angry_data_loader]
    val_data_loaders = [
        val_happy_data_loader, val_sad_data_loader, val_angry_data_loader
    ]
    tags = ['happy', 'sad', 'angry']

    for epoch in range(num_epochs):
        # train factual
        res = train_factual(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer,
                            criterion=criterion,
                            data_loader=data_loader,
                            log_step=log_step)

        val_res = val_factual(encoder=encoder,
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

        # train style
        res = train_emotion(encoder=encoder,
                            decoder=decoder,
                            optimizer=lang_optimizer,
                            criterion=criterion,
                            data_loaders=data_loaders,
                            tags=tags,
                            log_step=log_step_emotion)
        batch_time, losses = res
        val_res = val_emotion(encoder=encoder,
                              decoder=decoder,
                              vocab=vocab,
                              criterion=criterion,
                              data_loaders=val_data_loaders,
                              tags=tags)
        val_batch_time, top5accs, losses_val = val_res
        batch_time += val_batch_time
        print("""Batch Time: {:.3f}""".format(batch_time))
        for i in range(len(tags)):
            print(
                """Epoch [{}/{}], [{}], Batch Time: {:.3f}, Top-5 Acc: {:.3f}"""
                .format(epoch, num_epochs, tags[i][:3].upper(), batch_time,
                        top5accs[i]))
            print("""\tTrain Loss: {:.4f} | Train Perplexity: {:5.4f}""".format(
                losses[i], np.exp(losses[i])))
            print("""\tVal   Loss: {:.4f} | Val   Perplexity: {:5.4f}""".format(
                losses_val[i], np.exp(losses_val[i])))

        # Save the model checkpoints
        torch.save(
            decoder.state_dict(),
            os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
        torch.save(
            encoder.state_dict(),
            os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))


def val_factual(encoder, decoder, vocab, criterion, data_loader):
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
        lengths = [l - 1 for l in lengths]
        targets = pack_padded_sequence(input=captions[:, 1:],
                                       lengths=lengths,
                                       batch_first=True)[0]
        # Forward, backward and optimize
        features = encoder(images)
        outputs, alphas = decoder(captions[:, :-1],
                                  lengths,
                                  features,
                                  teacher_forcing_ratio=0)
        loss = criterion(outputs, targets)
        alpha_c = 1.
        loss += alpha_c * ((1. - alphas.sum(dim=1))**2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        top5 = accuracy(outputs, targets, 5)
        top5accs.update(top5, sum(lengths))
        batch_time.update(time.time() - start)

    feature = features[0].unsqueeze(0)
    start_token = vocab.word2idx['<start>']
    end_token = vocab.word2idx['<end>']
    sampled_ids = decoder.sample(feature,
                                 start_token=start_token,
                                 end_token=end_token)
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


def train_factual(encoder, decoder, optimizer, criterion, data_loader,
                  log_step):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        lengths = [l - 1 for l in lengths]
        targets = pack_padded_sequence(input=captions[:, 1:],
                                       lengths=lengths,
                                       batch_first=True)[0]
        # Forward, backward and optimize
        features = encoder(images)
        outputs, alphas = decoder(captions[:, :-1], lengths, features)
        loss = criterion(outputs, targets)
        alpha_c = 1.
        loss += alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
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


def val_emotion(encoder, decoder, vocab, criterion, data_loaders, tags):
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(tags))]
    top5accs = [AverageMeter() for _ in range(len(tags))]
    start = time.time()

    for j in range(len(tags)):
        for i, (images, captions, lengths) in enumerate(data_loaders[j]):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            lengths = [l - 1 for l in lengths]
            targets = pack_padded_sequence(input=captions[:, 1:],
                                           lengths=lengths,
                                           batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            outputs, alphas = decoder(captions[:, :-1],
                                      lengths,
                                      features,
                                      teacher_forcing_ratio=0,
                                      mode=tags[j])
            loss = criterion(outputs, targets)
            alpha_c = 1.
            loss += alpha_c * ((1. - alphas.sum(dim=1))**2).mean()

            # Keep track of metrics
            losses[j].update(loss.item(), sum(lengths))
            top5 = accuracy(outputs, targets, 5)
            top5accs[j].update(top5, sum(lengths))
            batch_time.update(time.time() - start)

        feature = features[0].unsqueeze(0)

        start_token = vocab.word2idx['<start>']
        end_token = vocab.word2idx['<end>']
        sampled_ids = decoder.sample(feature,
                                     start_token=start_token,
                                     end_token=end_token,
                                     mode=tags[j])
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break

        print(sampled_caption)

    top5accs = [top5acc.avg for top5acc in top5accs]
    losses = [loss.avg for loss in losses]
    return batch_time.val, top5accs, losses


def train_emotion(encoder, decoder, optimizer, criterion, data_loaders, tags,
                  log_step):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(tags))]
    start = time.time()

    for j in random.sample([i for i in range(len(tags))], len(tags)):
        for i, (images, captions, lengths) in enumerate(data_loaders[j]):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            lengths = [l - 1 for l in lengths]
            targets = pack_padded_sequence(input=captions[:, 1:],
                                           lengths=lengths,
                                           batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            outputs, alphas = decoder(captions[:, :-1],
                                      lengths,
                                      features,
                                      mode=tags[j])
            loss = criterion(outputs, targets)
            alpha_c = 1.
            loss += alpha_c * ((1. - alphas.sum(dim=1))**2).mean()

            decoder.zero_grad()
            # encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print("""Step [{}/{}], [{}], Loss: {:.4f}""".format(
                    i, len(data_loaders[j]), tags[j][:3].upper(), loss.item()))
            # Keep track of metrics
            losses[j].update(loss.item(), sum(lengths))
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
    parser.add_argument('--val_caption_path',
                        type=str,
                        default='data/flickr8k_id/val.txt',
                        help='path for val txt file')
    parser.add_argument('--happy_path',
                        type=str,
                        default='data/flickr8k_id/happy/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_happy_path',
                        type=str,
                        default='data/flickr8k_id/happy/val.txt',
                        help='path for train txt file')
    parser.add_argument('--sad_path',
                        type=str,
                        default='data/flickr8k_id/sad/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_sad_path',
                        type=str,
                        default='data/flickr8k_id/sad/val.txt',
                        help='path for train txt file')
    parser.add_argument('--angry_path',
                        type=str,
                        default='data/flickr8k_id/angry/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_angry_path',
                        type=str,
                        default='data/flickr8k_id/angry/val.txt',
                        help='path for train txt file')

    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--log_step_emotion', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--factored_size', type=int, default=512)
    parser.add_argument('--attention_size', type=int, default=512)
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
