import torch
import torch.nn as nn
import torchvision.models as models
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # delete the last fc layer.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class EncoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):

        super(EncoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, src_tokens, lengths):
        embeddings = self.embed(src_tokens)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, (h_t, c_t) = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs, (h_t, c_t)


class DecoderRNN(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, states, dst_tokens, lengths):
        embeddings = self.embed(dst_tokens)
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed, states)
        outputs = self.linear(hiddens[0])
        return outputs


class Seq2Seq(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 max_seq_length=40):
        super(Seq2Seq, self).__init__()
        self.max_seq_length = max_seq_length
        self.encoder = EncoderRNN(embed_size, hidden_size, vocab_size,
                                  num_layers)
        # happy
        self.decoder_happy = DecoderRNN(embed_size, hidden_size, vocab_size,
                                        num_layers)
        # sad
        self.decoder_sad = DecoderRNN(embed_size, hidden_size, vocab_size,
                                      num_layers)
        # angry
        self.decoder_angry = DecoderRNN(embed_size, hidden_size, vocab_size,
                                        num_layers)

    def forward(self, features, src, dst=(None, None), mode='factual'):
        src_tokens, src_lengths = src
        dst_tokens, dst_lengths = dst

        outputs, states = self.encoder(features, src_tokens, src_lengths)
        if mode == 'factual':
            return outputs

        if mode == 'happy':
            decoder = self.decoder_happy
        elif mode == 'sad':
            decoder = self.decoder_sad
        elif mode == 'angry':
            decoder = self.decoder_angry
        else:
            sys.stderr.write("mode name wrong!")

        outputs = decoder(states, dst_tokens, dst_lengths)

        return outputs
