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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchvision import transforms
from utils import AverageMeter, accuracy, adjust_learning_rate, clip_gradient, save_checkpoint
from nltk.translate.bleu_score import corpus_bleu
from copy import deepcopy
# from validate import validate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(0)

# resolve pytorch share multiprocess
torch.multiprocessing.set_sharing_strategy('file_system')

checkpoint_path = None


def main(args):
    log_path = args.log_path
    model_path = args.model_path
    crop_size = args.crop_size
    vocab_path = args.vocab_path
    num_workers = args.num_workers
    grad_clip = args.grad_clip

    image_dir = args.image_dir
    caption_path = args.caption_path
    val_caption_path = args.val_caption_path
    caption_batch_size = args.caption_batch_size

    embed_size = args.embed_size
    factored_size = args.factored_size
    hidden_size = args.hidden_size
    attention_size = args.attention_size
    dropout = args.dropout

    lr_caption = args.lr_caption
    num_epochs = args.num_epochs
    log_step = args.log_step

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
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

    # Loss
    criterion = nn.CrossEntropyLoss()

    if checkpoint_path is None:
        start_epoch = 0
        epochs_since_improvement = {'factual': 0, 'emotion': 0}
        best_bleu4 = {'factual': 0., 'emotion': 0.}

        # Build the models
        encoder = EncoderCNN().to(device)
        decoder = DecoderFactoredLSTMAtt(attention_size,
                                         embed_size,
                                         hidden_size,
                                         factored_size,
                                         len(vocab),
                                         1,
                                         dropout=dropout).to(device)
        # optimizer
        params = list(decoder.parameters()) + list(
            encoder.adaptive_pool.parameters())
        optimizer = torch.optim.Adam(params, lr=lr_caption)

    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
        optimizer = checkpoint['optimizer']
        print('start_epoch', start_epoch)

    # Train the models
    for epoch in range(start_epoch, num_epochs):

        # Decay learning rate if there is no improvement for 4 consecutive epochs, and terminate training after 10
        imp_fac = epochs_since_improvement['factual']
        if imp_fac >= 10:
            break
        if imp_fac > 0 and imp_fac % 4 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train factual
        res = train_factual(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer,
                            criterion=criterion,
                            data_loader=data_loader,
                            log_step=log_step,
                            grad_clip=grad_clip)

        val_res = val_factual(encoder=encoder,
                              decoder=decoder,
                              vocab=vocab,
                              criterion=criterion,
                              data_loader=val_data_loader)
        batch_time, loss = res
        val_batch_time, top5, loss_val, bleu4 = val_res
        batch_time += val_batch_time
        text = """Epoch [{}/{}], [FAC], Batch Time: {:.3f}, Top-5 Acc: {:.3f}, BLEU-4 Score: {}\n""".format(
            epoch, num_epochs, batch_time, top5, bleu4)
        text += """\tTrain Loss: {:.4f} | Train Perplexity: {:5.4f}\n""".format(
            loss, np.exp(loss))
        text += """\tVal   Loss: {:.4f} | Val   Perplexity: {:5.4f}\n""".format(
            loss_val, np.exp(loss_val))
        print(text)
        open(log_path, 'a+').write(text)

        is_best = bleu4 > best_bleu4['factual']
        best_bleu4['factual'] = max(bleu4, best_bleu4['factual'])
        if not is_best:
            epochs_since_improvement['factual'] += 1
            print("Epochs [FAC] since last improvement:",
                  epochs_since_improvement['factual'])
        else:
            epochs_since_improvement['factual'] = 0

        # Save the model checkpoints
        save_checkpoint('models', model_path, 'FAC', epoch,
                        epochs_since_improvement, encoder, decoder, optimizer,
                        None, best_bleu4, is_best)


def val_factual(encoder, decoder, vocab, criterion, data_loader):
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    # references (true captions) for calculating BLEU-4 score
    references = list()
    # hypotheses (predictions)
    hypotheses = list()

    for i, (images, captions, lengths, all_captions) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        lengths = [l - 1 for l in lengths]
        packed_targets = pack_padded_sequence(input=captions[:, 1:],
                                              lengths=lengths,
                                              batch_first=True)
        targets = packed_targets.data
        # Forward, backward and optimize
        with torch.no_grad():
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

        # unpacked outputs
        scores = outputs.clone()
        scores = PackedSequence(scores, packed_targets.batch_sizes)
        scores = pad_packed_sequence(scores, batch_first=True)

        start = vocab.word2idx['<start>']
        end = vocab.word2idx['<end>']
        all_caps = deepcopy(all_captions)
        for caps in all_caps:
            caps = [c.long().tolist() for c in caps]
            caps = [[w for w in c if w != start and w != end] for c in caps]
            references.append(caps)

        preds = list()
        for s, l in zip(scores[0], scores[1]):
            _, pred = torch.max(s, dim=1)
            pred = pred.tolist()[:l]
            pred = [w for w in pred if w != start and w != end]
            preds.append(pred)
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)
        # free
        del images
        del captions
        del lengths
        del all_captions
        del packed_targets
        del outputs
        del alphas

    torch.cuda.empty_cache()

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

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

    return batch_time.val, top5accs.avg, losses.avg, bleu4


def train_factual(encoder, decoder, optimizer, criterion, data_loader, log_step,
                  grad_clip):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, (images, captions, lengths, all_captions) in enumerate(data_loader):
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
        # Clip gradients
        clip_gradient(optimizer, grad_clip)
        optimizer.step()

        if i % log_step == 0:
            print("""Step [{}/{}], [FAC], Loss: {:.4f}""".format(
                i, len(data_loader), loss.item()))

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        batch_time.update(time.time() - start)
        # free
        del images
        del captions
        del lengths
        del all_captions
        del targets
        del outputs
        del alphas

    torch.cuda.empty_cache()

    return batch_time.val, losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related
    parser.add_argument('--log_path',
                        type=str,
                        default='out.log',
                        help='path for logging')
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

    parser.add_argument('--log_step', type=int, default=50)
    # parser.add_argument('--log_step_emotion', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--factored_size', type=int, default=512)
    parser.add_argument('--attention_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--caption_batch_size', type=int, default=64)
    # parser.add_argument('--language_batch_size', type=int, default=96)
    parser.add_argument('--lr_caption', type=float, default=0.0002)
    # parser.add_argument('--lr_language', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
