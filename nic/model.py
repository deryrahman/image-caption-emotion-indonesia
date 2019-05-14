import torch
import torch.nn as nn
import torchvision.models as models

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


class DecoderRNN(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 max_seq_length=40):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(inputs, states)

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
