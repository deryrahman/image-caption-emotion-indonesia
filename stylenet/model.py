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


class DecoderFactoredLSTM(nn.Module):

    def __init__(self,
                 embed_size,
                 hidden_size,
                 factored_size,
                 vocab_size,
                 num_layers,
                 bias=True,
                 max_seq_length=40):
        super(DecoderFactoredLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # embedding
        self.B = nn.Embedding(vocab_size, embed_size)

        # factored lstm weights
        self.U_i = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fi = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_i = nn.Linear(embed_size, factored_size, bias=bias)
        self.W_i = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_f = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_ff = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_f = nn.Linear(embed_size, factored_size, bias=bias)
        self.W_f = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_o = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fo = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_o = nn.Linear(embed_size, factored_size, bias=bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_c = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fc = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_c = nn.Linear(embed_size, factored_size, bias=bias)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=bias)

        # happy
        self.S_happy_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_c = nn.Linear(factored_size, factored_size, bias=bias)

        # sad
        self.S_sad_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_c = nn.Linear(factored_size, factored_size, bias=bias)

        # angry
        self.S_angry_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_c = nn.Linear(factored_size, factored_size, bias=bias)

        # weight for output
        self.C = nn.Linear(hidden_size, vocab_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward_step(self, embedded, states, mode):
        batch_size = embedded.size(0)
        h_t, c_t = states
        if h_t is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        if c_t is None:
            c_t = torch.zeros(batch_size, self.hidden_size).to(device)

        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)

        if mode == "factual":
            i = self.S_fi(i)
            f = self.S_ff(f)
            o = self.S_fo(o)
            c = self.S_fc(c)
        elif mode == "happy":
            i = self.S_happy_i(i)
            f = self.S_happy_f(f)
            o = self.S_happy_o(o)
            c = self.S_happy_c(c)
        elif mode == "sad":
            i = self.S_sad_i(i)
            f = self.S_sad_f(f)
            o = self.S_sad_o(o)
            c = self.S_sad_c(c)
        elif mode == "angry":
            i = self.S_angry_i(i)
            f = self.S_angry_f(f)
            o = self.S_angry_o(o)
            c = self.S_angry_c(c)
        else:
            sys.stderr.write("mode name wrong!")

        i_t = torch.sigmoid(self.U_i(i) + self.W_i(h_t))
        f_t = torch.sigmoid(self.U_f(f) + self.W_f(h_t))
        o_t = torch.sigmoid(self.U_o(o) + self.W_o(h_t))
        c_tilda = torch.tanh(self.U_c(c) + self.W_c(h_t))

        c_t = f_t * c_t + i_t * c_tilda
        h_t = o_t * c_t

        return h_t, (h_t, c_t)

    def forward(self,
                captions,
                lengths,
                features=None,
                teacher_forcing_ratio=0.5,
                mode='factual'):
        batch_size = captions.size(0)
        embeddings = self.B(captions)

        # concat features and captions
        if features is not None:
            embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)
        hiddens = []
        predicted = captions[:, 0:1]
        for i, b_sz in enumerate(packed.batch_sizes):
            if random.random() < teacher_forcing_ratio:
                emb = embeddings[:b_sz, i, :]
            else:
                emb = self.B(predicted)[:b_sz, 0, :]
            h_t, c_t = h_t[:b_sz, :], c_t[:b_sz, :]
            hidden, (h_t, c_t) = self.forward_step(emb, (h_t, c_t), mode=mode)
            hiddens.append(hidden)

            output = self.C(hidden)
            _, predicted = output.max(1)
            predicted = predicted.unsqueeze(1)

        hiddens = torch.cat(hiddens, 0)
        outputs = self.C(hiddens)

        return outputs

    def sample(self, features, states=None, factual_limit=-1, mode='factual'):
        """Generate captions for given image features using greedy search."""
        states = (None, None)
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_seq_length):
            # hiddens: (batch_size, 1, hidden_size)
            m = mode if i > factual_limit else 'factual'
            hiddens, states = self.forward_step(inputs, states, m)

            # outputs:  (batch_size, vocab_size)
            outputs = self.C(hiddens.squeeze(1))

            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, embed_size)
            inputs = self.B(predicted)

            # inputs: (batch_size, 1, embed_size)
            inputs = inputs.unsqueeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)

        return sampled_ids
