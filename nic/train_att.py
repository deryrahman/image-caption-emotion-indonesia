import argparse
import torch
import torch.nn as nn
import numpy as np
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
# from validate import validate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(0)

# resolve pytorch share multiprocess
torch.multiprocessing.set_sharing_strategy('file_system')

checkpoint_path = './models/FAC_BEST_checkpoint_nic_att_30k_4.pth.tar'
is_fac = True


def main(args):
    log_path = args.log_path
    model_path = args.model_path
    crop_size = args.crop_size
    vocab_path = args.vocab_path
    num_workers = args.num_workers
    grad_clip = args.grad_clip

    mode = args.mode
    image_dir = args.image_dir
    val_emotion_path = args.emotion_path

    emotion_path = args.emotion_path
    language_batch_size = args.language_batch_size

    lr_language = args.lr_language
    num_epochs = args.num_epochs
    log_step_emotion = args.log_step_emotion

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

    # Loss
    criterion = nn.CrossEntropyLoss()

    if is_fac:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = 0
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        curr_bleu4 = checkpoint['bleu-4']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
        optimizer = checkpoint['optimizer']
        lang_params = list(decoder.parameters())
        lang_optimizer = torch.optim.Adam(lang_params, lr=lr_language)
    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        curr_bleu4 = checkpoint['bleu-4']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
        optimizer = checkpoint['optimizer']
        lang_optimizer = checkpoint['lang_optimizer']
        print('start_epoch', start_epoch)

    # Train the models
    for epoch in range(start_epoch, num_epochs):

        # Decay learning rate if there is no improvement for 4 consecutive epochs, and terminate training after 10
        imp_emo = epochs_since_improvement['emotion']
        if imp_emo >= 10:
            break
        if imp_emo > 0 and imp_emo % 4 == 0:
            adjust_learning_rate(lang_optimizer, 0.8)

        # train style
        res = train_emotion(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer,
                            criterion=criterion,
                            data_loader=emotion_data_loader,
                            log_step=log_step_emotion,
                            grad_clip=grad_clip,
                            mode=mode)

        val_res = val_emotion(encoder=encoder,
                              decoder=decoder,
                              vocab=vocab,
                              criterion=criterion,
                              data_loader=val_emotion_data_loader)
        batch_time, loss = res
        val_batch_time, top5, loss_val, bleu4 = val_res
        batch_time += val_batch_time
        text = """Epoch [{}/{}], [{}], Batch Time: {:.3f}, Top-5 Acc: {:.3f}, BLEU-4 Score: {}\n""".format(
            epoch, num_epochs, mode[:3].upper(), batch_time, top5, bleu4)
        text += """\tTrain Loss: {:.4f} | Train Perplexity: {:5.4f}\n""".format(
            loss, np.exp(loss))
        text += """\tVal   Loss: {:.4f} | Val   Perplexity: {:5.4f}\n""".format(
            loss_val, np.exp(loss_val))
        print(text)
        open(log_path, 'a+').write(text)

        is_best = bleu4 > best_bleu4['emotion']
        best_bleu4['emotion'] = max(bleu4, best_bleu4['emotion'])
        if not is_best:
            epochs_since_improvement['emotion'] += 1
            print("Epochs [{}] since last improvement: {}".format(
                mode[:3].upper(), epochs_since_improvement['emotion']))
        else:
            epochs_since_improvement['emotion'] = 0
        curr_bleu4['emotion'] = bleu4

        # Save the model checkpoints
        save_checkpoint('models', model_path, mode[:3].upper(), epoch,
                        epochs_since_improvement, encoder, decoder, optimizer,
                        lang_optimizer, curr_bleu4, is_best)


def val_emotion(encoder, decoder, vocab, criterion, data_loader):
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
        for caps in all_captions:
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


def train_emotion(encoder, decoder, optimizer, criterion, data_loader, log_step,
                  grad_clip, mode):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, (images, captions, lengths, _) in enumerate(data_loader):
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
            print("""Step [{}/{}], [{}], Loss: {:.4f}""".format(
                i, len(data_loader), mode[:3].upper(), loss.item()))

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        batch_time.update(time.time() - start)

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
    parser.add_argument('--mode', type=str, default='happy')
    parser.add_argument('--vocab_path',
                        type=str,
                        default='data/flickr8k_id/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir',
                        type=str,
                        default='/home/m13515097/final_project/'
                        '13515097-stylenet/dataset/flickr30k/img',
                        help='directory for train images')
    parser.add_argument('--emotion_path',
                        type=str,
                        default='data/flickr8k_id/happy/train.txt',
                        help='path for train txt file')
    parser.add_argument('--val_emotion_path',
                        type=str,
                        default='data/flickr8k_id/happy/val.txt',
                        help='path for train txt file')

    # parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--log_step_emotion', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # # Model parameters (already define in factual)
    # parser.add_argument('--embed_size', type=int, default=300)
    # parser.add_argument('--hidden_size', type=int, default=512)
    # parser.add_argument('--attention_size', type=int, default=512)
    # parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=120)
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--caption_batch_size', type=int, default=64)
    parser.add_argument('--language_batch_size', type=int, default=96)
    # parser.add_argument('--lr_caption', type=float, default=0.0002)
    parser.add_argument('--lr_language', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)
