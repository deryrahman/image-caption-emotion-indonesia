import torch
import torch.nn as nn
import torchvision.models as models
import sys
import random

random.seed(0)
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

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 dropout=0.22,
                 max_seq_length=40):

        super(EncoderRNN, self).__init__()
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, embedded, states):
        batch_size = embedded.size(0)
        h_t, c_t = states
        if h_t is None:
            h_t = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(device)
        if c_t is None:
            c_t = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(device)

        if len(embedded.size()) == 2:
            embedded = embedded.unsqueeze(1)
        hiddens, (h_t, c_t) = self.lstm(embedded, (h_t, c_t))
        hiddens = hiddens.squeeze(1)
        return hiddens, (h_t, c_t)

    def forward(self, features, src_tokens, lengths, teacher_forcing_ratio=0.5):
        batch_size = src_tokens.size(0)
        embeddings = self.embed(src_tokens)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        h_t = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(device)
        c_t = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(device)
        hiddens = []
        predicted = src_tokens[:, 0:1]
        for i, b_sz in enumerate(packed.batch_sizes):
            if random.random() < teacher_forcing_ratio:
                emb = embeddings[:b_sz, i, :]
            else:
                emb = self.embed(predicted)[:b_sz, 0, :]
            h_t, c_t = h_t[:, :b_sz, :], c_t[:, :b_sz, :]
            hidden, (h_t, c_t) = self.forward_step(emb, (h_t, c_t))
            hiddens.append(hidden)

            output = self.linear(hidden)
            _, predicted = output.max(1)
            predicted = predicted.unsqueeze(1)

        hiddens = torch.cat(hiddens, 0)
        outputs = self.linear(hiddens)

        return outputs, (h_t, c_t)

    def sample(self, features, states=(None, None)):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.forward_step(inputs, states)

            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))

            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)

            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids, states


class DecoderRNN(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 dropout=0.22,
                 max_seq_length=40):
        super(DecoderRNN, self).__init__()
        self.max_seq_length = max_seq_length
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, embedded, states):
        batch_size = embedded.size(0)
        h_t, c_t = states
        if h_t is None:
            h_t = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(device)
        if c_t is None:
            c_t = torch.zeros(self.num_layers, batch_size,
                              self.hidden_size).to(device)

        if len(embedded.size()) == 2:
            embedded = embedded.unsqueeze(1)
        hiddens, (h_t, c_t) = self.lstm(embedded, (h_t, c_t))
        hiddens = hiddens.squeeze(1)
        return hiddens, (h_t, c_t)

    def forward(self, states, dst_tokens, lengths, teacher_forcing_ratio=0.5):
        batch_size = dst_tokens.size(0)
        embeddings = self.embed(dst_tokens)
        embeddings = self.dropout(embeddings)
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        h_t = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(device)
        c_t = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(device)
        hiddens = []
        predicted = dst_tokens[:, 0:1]
        for i, b_sz in enumerate(packed.batch_sizes):
            if random.random() < teacher_forcing_ratio:
                emb = embeddings[:b_sz, i, :]
            else:
                emb = self.embed(predicted)[:b_sz, 0, :]
            h_t, c_t = h_t[:, :b_sz, :], c_t[:, :b_sz, :]
            hidden, (h_t, c_t) = self.forward_step(emb, (h_t, c_t))
            hiddens.append(hidden)

            output = self.linear(hidden)
            _, predicted = output.max(1)
            predicted = predicted.unsqueeze(1)

        hiddens = torch.cat(hiddens, 0)
        outputs = self.linear(hiddens)

        return outputs

    def sample(self, start_token, states):
        sampled_ids = []
        inputs = torch.zeros(1, 1)
        inputs = torch.Tensor([[start_token]]).long().to(device)
        inputs = self.embed(inputs)
        for i in range(self.max_seq_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.forward_step(inputs, states)

            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))

            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, embed_size)
            inputs = self.embed(predicted)

            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class Seq2Seq(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 dropout=0.22,
                 max_seq_length=40):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.encoder = EncoderRNN(embed_size,
                                  hidden_size,
                                  vocab_size,
                                  num_layers,
                                  dropout=dropout)
        # happy
        self.decoder_happy = DecoderRNN(embed_size,
                                        hidden_size,
                                        vocab_size,
                                        num_layers,
                                        dropout=dropout)
        # sad
        self.decoder_sad = DecoderRNN(embed_size,
                                      hidden_size,
                                      vocab_size,
                                      num_layers,
                                      dropout=dropout)
        # angry
        self.decoder_angry = DecoderRNN(embed_size,
                                        hidden_size,
                                        vocab_size,
                                        num_layers,
                                        dropout=dropout)

    def forward(self,
                features,
                src,
                dst=(None, None),
                teacher_forcing_ratio=0.8,
                mode='factual'):
        src_tokens, src_lengths = src
        dst_tokens, dst_lengths = dst

        outputs, states = self.encoder(features, src_tokens, src_lengths,
                                       teacher_forcing_ratio)

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

        outputs = decoder(states, dst_tokens, dst_lengths,
                          teacher_forcing_ratio)

        return outputs

    def sample(self, features, start_token, states=(None, None),
               mode='factual'):
        sampled_ids, states = self.encoder.sample(features, states)
        if mode == 'factual':
            return sampled_ids

        if mode == 'happy':
            decoder = self.decoder_happy
        elif mode == 'sad':
            decoder = self.decoder_sad
        elif mode == 'angry':
            decoder = self.decoder_angry
        else:
            sys.stderr.write("mode name wrong!")

        sampled_ids = decoder.sample(start_token, states)
        return sampled_ids
