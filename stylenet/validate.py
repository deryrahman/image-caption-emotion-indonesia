from nltk.translate.bleu_score import corpus_bleu
from utils import accuracy, AverageMeter
from torch.nn.utils.rnn import pack_padded_sequence
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(val_loader, log_step, word_map, encoder, decoder, criterion):
    # eval mode (no dropout or batchnorm)
    decoder.eval()
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()
    images_captions = {}

    for i, (images, captions, lengths) in enumerate(val_loader):

        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(captions, lengths, features)
        scores = outputs
        loss = criterion(outputs, targets)

        # Keep track of metrics
        losses.update(loss.item(), sum(lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % log_step == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top5=top5accs))

        outputs = []
        for feature in features:
            outputs.append(decoder.sample(feature.unsqueeze(0))[0])
        for image, target, predict in zip(images, captions, outputs):
            predict = predict.cpu().numpy()
            target = target.cpu().numpy()
            sampled_caption = []
            for word_id in predict:
                word = word_map[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            predict = sampled_caption[1:-1]

            sampled_caption = []
            for word_id in target:
                word = word_map[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            target = sampled_caption[1:-1]

            if images_captions.get(image) is None:
                images_captions[image] = {'ref': [], 'pred': []}
            images_captions[image]['ref'].append(target)
            images_captions[image]['pred'].append(predict)

    print('[SAMPLED]')
    print('Predict:', predict)
    print('Target:', target)
    references = [cap['ref'] * 5 for cap in images_captions.values()]
    hypotheses = [
        pred for cap in images_captions.values() for pred in cap['pred']
    ]

    assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'
        .format(loss=losses, top5=top5accs, bleu=bleu4))

    return bleu4
