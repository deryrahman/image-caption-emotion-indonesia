import torch
import torch.nn as nn
import torchvision.models as models
import random

random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):

    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # delete the last fc layer and pool layer.
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = self.adaptive_pool(features)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        features = features.permute(0, 2, 3, 1)
        return features


class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        # (batch_size, num_pixels)
        alpha = self.softmax(att)
        # (batch_size, encoder_dim)
        w = encoder_out * alpha.unsqueeze(2)
        attention_weighted_encoding = (w).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderRNNAtt(nn.Module):

    def __init__(self,
                 attention_size,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 feature_size=2048,
                 dropout=0.22,
                 max_seq_length=40):
        super(DecoderRNNAtt, self).__init__()
        self.attention_size = attention_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        # linear layer to find initial hidden state
        self.init_h = nn.Linear(feature_size, hidden_size)
        # linear layer to find initial cell state
        self.init_c = nn.Linear(feature_size, hidden_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # attention network
        self.attention = Attention(feature_size, hidden_size, attention_size)

        # embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_size, feature_size)
        self.sigmoid = nn.Sigmoid()

        # lstm
        self.lstm = nn.LSTMCell(embed_size + feature_size,
                                hidden_size,
                                bias=True)

        # weight for output
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.reset_parameters()
        self.init_weights()

    def reset_parameters(self):
        # std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, feature):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param feature: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_feature = feature.mean(dim=1)
        h = self.init_h(mean_feature)  # (batch_size, decoder_dim)
        c = self.init_c(mean_feature)
        return h, c

    def forward_step(self, embedded, states):
        h_t, c_t = states

        h_t, c_t = self.lstm(embedded, (h_t, c_t))

        return h_t, (h_t, c_t)

    def forward(self, captions, lengths, features, teacher_forcing_ratio=0.8):
        batch_size = captions.size(0)
        feature_size = features.size(-1)

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        features = features.view(batch_size, -1, feature_size)
        num_pixels = features.size(1)

        # embeddings
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)

        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        # (batch_size, decoder_dim)
        h_t, c_t = self.init_hidden_state(features)
        alphas = torch.zeros(batch_size, max(lengths), num_pixels).to(device)

        hiddens = []
        predicted = captions[:, 0:1]
        for i, b_sz in enumerate(packed.batch_sizes):

            attention_weighted_encoding, alpha = self.attention(
                features[:b_sz], h_t[:b_sz])

            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h_t[:b_sz]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            if random.random() < teacher_forcing_ratio:
                emb = embeddings[:b_sz, i, :]
            else:
                emb = self.embed(predicted)[:b_sz, 0, :]

            inputs = torch.cat([emb, attention_weighted_encoding], dim=1)
            h_t, c_t = h_t[:b_sz, :], c_t[:b_sz, :]
            hidden, (h_t, c_t) = self.forward_step(inputs, (h_t, c_t))

            hiddens.append(hidden)
            alphas[:b_sz, i, :] = alpha

            output = self.linear(hidden)
            _, predicted = output.max(1)
            predicted = predicted.unsqueeze(1)

        hiddens = torch.cat(hiddens, 0)
        outputs = self.linear(hiddens)

        return outputs, alphas

    def sample(self, features, start_token, end_token, k=5):
        """Generate captions for given image features using beam search."""

        # enc_image_size = features.size(1)
        feature_size = features.size(-1)
        # batch_size = features.size(0)

        features = features.view(1, -1, feature_size)
        num_pixels = features.size(1)
        # (k, num_pixels, encoder_dim)
        features = features.expand(k, num_pixels, feature_size)
        # (k, 1)
        k_prev_words = torch.LongTensor([[start_token]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h_t, c_t = self.init_hidden_state(features)

        while True:
            # (s, embed_dim)
            embeddings = self.embed(k_prev_words).squeeze(1)
            # (s, encoder_dim), (s, num_pixels)
            awe, _ = self.attention(features, h_t)

            # gating scalar, (s, encoder_dim)
            gate = self.sigmoid(self.f_beta(h_t))
            awe = gate * awe

            inputs = torch.cat([embeddings, awe], dim=1)
            res = self.forward_step(inputs, (h_t, c_t))
            hidden, (h_t, c_t) = res

            # (s, vocab_size)
            output = self.linear(hidden)
            scores = torch.nn.functional.log_softmax(output, dim=1)

            # Add
            # (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                # (s)
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(-1).topk(
                    k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences
            # (s, step+1)
            seqs = torch.cat(
                [seqs[prev_word_inds],
                 next_word_inds.unsqueeze(1)], dim=1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds)
                if next_word != end_token
            ]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            # reduce beam length accordingly
            k -= len(complete_inds)

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h_t = h_t[prev_word_inds[incomplete_inds]]
            c_t = c_t[prev_word_inds[incomplete_inds]]
            features = features[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > self.max_seq_length:
                break
            step += 1

        # prevent empty sequence
        if len(complete_seqs_scores) == 0:
            return torch.Tensor([[end_token]]).long().to(device)

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = torch.Tensor([complete_seqs[i]]).long().to(device)

        return seq
