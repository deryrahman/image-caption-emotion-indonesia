import torch
import torch.nn as nn
import torchvision.models as models
import sys
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


class DecoderFactoredLSTMAtt(nn.Module):

    def __init__(self,
                 attention_size,
                 embed_size,
                 hidden_size,
                 factored_size,
                 vocab_size,
                 num_layers,
                 feature_size=2048,
                 bias=True,
                 dropout=0.22,
                 max_seq_length=40):
        super(DecoderFactoredLSTMAtt, self).__init__()
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
        self.B = nn.Embedding(vocab_size, embed_size)

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_size, feature_size)
        self.sigmoid = nn.Sigmoid()

        # factored lstm weights
        self.U_i = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fi = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_i = nn.Linear(embed_size + feature_size,
                             factored_size,
                             bias=bias)
        self.W_i = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_f = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_ff = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_f = nn.Linear(embed_size + feature_size,
                             factored_size,
                             bias=bias)
        self.W_f = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_o = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fo = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_o = nn.Linear(embed_size + feature_size,
                             factored_size,
                             bias=bias)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.U_c = nn.Linear(factored_size, hidden_size, bias=bias)
        self.S_fc = nn.Linear(factored_size, factored_size, bias=bias)
        self.V_c = nn.Linear(embed_size + feature_size,
                             factored_size,
                             bias=bias)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=bias)

        # happy
        self.attention_happy = Attention(feature_size, hidden_size,
                                         attention_size)
        self.S_happy_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_happy_c = nn.Linear(factored_size, factored_size, bias=bias)

        # sad
        self.attention_sad = Attention(feature_size, hidden_size,
                                       attention_size)
        self.S_sad_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_sad_c = nn.Linear(factored_size, factored_size, bias=bias)

        # angry
        self.attention_angry = Attention(feature_size, hidden_size,
                                         attention_size)
        self.S_angry_i = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_f = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_o = nn.Linear(factored_size, factored_size, bias=bias)
        self.S_angry_c = nn.Linear(factored_size, factored_size, bias=bias)

        # weight for output
        self.C = nn.Linear(hidden_size, vocab_size, bias=bias)

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
        self.B.weight.data.uniform_(-0.1, 0.1)
        self.C.bias.data.fill_(0)
        self.C.weight.data.uniform_(-0.1, 0.1)

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

    def forward_step(self, embedded, states, mode):
        # batch_size = embedded.size(0)
        h_t, c_t = states

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
                features,
                teacher_forcing_ratio=0.8,
                mode='factual'):
        batch_size = captions.size(0)
        feature_size = features.size(-1)

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        features = features.view(batch_size, -1, feature_size)
        num_pixels = features.size(1)

        # embeddings
        embeddings = self.B(captions)
        embeddings = self.dropout(embeddings)

        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        # (batch_size, decoder_dim)
        h_t, c_t = self.init_hidden_state(features)
        alphas = torch.zeros(batch_size, max(lengths), num_pixels).to(device)

        hiddens = []
        predicted = captions[:, 0:1]

        if mode == 'factual':
            attention = self.attention
        if mode == 'happy':
            attention = self.attention_happy
        elif mode == 'sad':
            attention = self.attention_sad
        elif mode == 'angry':
            attention = self.attention_angry
        else:
            sys.stderr.write("mode name wrong!")

        for i, b_sz in enumerate(packed.batch_sizes):

            attention_weighted_encoding, alpha = attention(
                features[:b_sz], h_t[:b_sz])

            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h_t[:b_sz]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            if random.random() < teacher_forcing_ratio:
                emb = embeddings[:b_sz, i, :]
            else:
                emb = self.B(predicted)[:b_sz, 0, :]

            inputs = torch.cat([emb, attention_weighted_encoding], dim=1)
            h_t, c_t = h_t[:b_sz, :], c_t[:b_sz, :]
            res = self.forward_step(inputs, (h_t, c_t), mode=mode)
            hidden, (h_t, c_t) = res

            hiddens.append(hidden)
            alphas[:b_sz, i, :] = alpha

            output = self.C(hidden)
            _, predicted = output.max(1)
            predicted = predicted.unsqueeze(1)

        hiddens = torch.cat(hiddens, 0)
        outputs = self.C(hiddens)

        return outputs, alphas

    def sample(self,
               features,
               start_token,
               end_token,
               k=5,
               factual_limit=-1,
               mode='factual'):
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

        if mode == 'factual':
            attention = self.attention
        elif mode == 'happy':
            attention = self.attention_happy
        elif mode == 'sad':
            attention = self.attention_sad
        elif mode == 'angry':
            attention = self.attention_angry
        else:
            sys.stderr.write("mode name wrong!")

        while True:
            # (s, embed_dim)
            embeddings = self.B(k_prev_words).squeeze(1)
            # (s, encoder_dim), (s, num_pixels)
            awe, _ = attention(features, h_t)

            # gating scalar, (s, encoder_dim)
            gate = self.sigmoid(self.f_beta(h_t))
            awe = gate * awe

            inputs = torch.cat([embeddings, awe], dim=1)
            res = self.forward_step(inputs, (h_t, c_t), mode=mode)
            hidden, (h_t, c_t) = res

            # (s, vocab_size)
            output = self.C(hidden)
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
