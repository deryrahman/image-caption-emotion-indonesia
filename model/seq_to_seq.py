from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import np
from loss import sparse_cross_entropy
from nn import FactoredLSTM, AttentionLayer
from model.base import RichModel
import tensorflow as tf
import os


class Seq2Seq(RichModel):

    def __init__(self,
                 mode,
                 with_attention=False,
                 trainable_model=False,
                 injection_mode='init',
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128,
                 factored_size=256,
                 learning_rate=0.0002,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 encoder_lstm_layers=1,
                 decoder_lstm_layers=1):
        self.mode = mode
        self.with_attention = with_attention
        self.injection_mode = injection_mode
        self.trainable_model = trainable_model
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.factored_size = factored_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.encoder_lstm_layers = encoder_lstm_layers
        self.decoder_lstm_layers = decoder_lstm_layers
        self.model = None
        self.model_partial = None
        self.model_encoder = None
        self.model_encoder_partial = None
        self.model_decoder = None
        self._build()

    def _build(self):
        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,),
            name='seq2seq_transfer_values_input')
        encoder_units = {
            'init': self.state_size,
            'pre': self.embedding_size
        }.get(self.injection_mode, self.state_size)
        encoder_transfer_map = Dense(
            encoder_units,
            activation='tanh',
            name='seq2seq_encoder_transfer_map',
            trainable=self.trainable_model)

        # encoder word embedding
        encoder_input = Input(shape=(None,), name='seq2seq_encoder_input')
        encoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='seq2seq_encoder_embedding',
            trainable=self.trainable_model)

        # encoder Factored LSTM
        encoder_factored_lstm = [
            FactoredLSTM(
                self.state_size,
                mode=self.mode,
                trainable_model=self.trainable_model,
                factored_dim=self.factored_size,
                name='seq2seq_encoder_factored_lstm_{}'.format(i),
                return_sequences=True,
                return_state=True) for i in range(self.encoder_lstm_layers)
        ]

        # decoder word embedding
        decoder_input = Input(shape=(None,), name='seq2seq_decoder_input')
        decoder_input_h = Input(
            shape=(self.state_size,), name='seq2seq_decoder_input_h')
        decoder_input_c = Input(
            shape=(self.state_size,), name='seq2seq_decoder_input_c')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='seq2seq_decoder_embedding')

        # decoder LSTM
        decoder_lstm = [
            LSTM(
                self.state_size,
                name='seq2seq_decoder_lstm_{}'.format(i),
                return_sequences=True,
                return_state=True) for i in range(self.decoder_lstm_layers)
        ]
        decoder_dense = Dense(
            self.num_words, activation='linear', name='seq2seq_decoder_output')

        # attention layer
        attention = AttentionLayer(name='seq2seq_attention')

        def connect_lstm(states, uniform_state, lstm_layers, net):

            for i in range(len(lstm_layers)):
                net, state_h, state_c = lstm_layers[i](
                    net, initial_state=states)

                if not uniform_state:
                    states = [state_h, state_c]

            return net, state_h, state_c

        def connect_encoder(transfer_values_input, encoder_input):

            encoder_net = encoder_embedding(encoder_input)

            if transfer_values_input is None:
                states = None
                encoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=False,
                    lstm_layers=encoder_factored_lstm,
                    net=encoder_net)
                return encoder_net, state_h, state_c

            encoder_transfer = encoder_transfer_map(transfer_values_input)

            if self.injection_mode == 'init':
                states = [encoder_transfer, encoder_transfer]
                encoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=True,
                    lstm_layers=encoder_factored_lstm,
                    net=encoder_net)
                return encoder_net, state_h, state_c

            if self.injection_mode == 'pre':
                states = None
                encoder_init = RepeatVector(1)(encoder_transfer)
                encoder_net = Concatenate(axis=1)([encoder_init, encoder_net])
                encoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=False,
                    lstm_layers=encoder_factored_lstm,
                    net=encoder_net)

                return encoder_net, state_h, state_c

            return None, None, None

        def connect_decoder(encoder_states, decoder_input):
            decoder_net = decoder_embedding(decoder_input)
            decoder_net, state_h, state_c = connect_lstm(
                states=encoder_states,
                uniform_state=True,
                lstm_layers=decoder_lstm,
                net=decoder_net)

            return decoder_net, state_h, state_c

        # connect full model
        encoder_net, state_h, state_c = connect_encoder(transfer_values_input,
                                                        encoder_input)
        decoder_net, _, _ = connect_decoder([state_h, state_c], decoder_input)

        if self.with_attention:
            att_net, attn_states = attention([encoder_net, decoder_net])
            decoder_net = Concatenate(axis=-1)([decoder_net, att_net])

        decoder_output = decoder_dense(decoder_net)
        self.model = Model(
            inputs=[transfer_values_input, encoder_input, decoder_input],
            outputs=[decoder_output])

        # connect full model without transfer value
        encoder_net, state_h, state_c = connect_encoder(None, encoder_input)
        decoder_net, _, _ = connect_decoder([state_h, state_c], decoder_input)

        if self.with_attention:
            att_net, attn_states = attention([encoder_net, decoder_net])
            decoder_net = Concatenate(axis=-1)([decoder_net, att_net])

        decoder_output = decoder_dense(decoder_net)
        self.model_partial = Model(
            inputs=[encoder_input, decoder_input], outputs=[decoder_output])

        # connect encoder FactoredLSTM
        _, state_h, state_c = connect_encoder(transfer_values_input,
                                              encoder_input)
        states = [state_h, state_c]
        self.model_encoder = Model(
            inputs=[transfer_values_input, encoder_input], outputs=states)

        # connect encoder FactoredLSTM without transfer value
        _, state_h, state_c = connect_encoder(None, encoder_input)
        states = [state_h, state_c]
        self.model_encoder_partial = Model(
            inputs=[encoder_input], outputs=states)

        # connect decoder LSTM
        states = [decoder_input_h, decoder_input_c]
        decoder_net, _, _ = connect_decoder(states, decoder_input)
        decoder_output = decoder_dense(decoder_net)
        self.model_decoder = Model(
            inputs=[decoder_input] + states, outputs=[decoder_output])

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        # model compile
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=decoder_target)
        self.model_partial.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=decoder_target)

    def save(self, path, overwrite):
        """Rewrite save model, only save weight from decoder and kernel_S encoder

        Arguments:
            path {str} -- string path to save model
            overwrite {bool} -- overwrite existing model or not
        """
        self.model_decoder.save_weights(path, overwrite=overwrite)

        # weights values for kernel_S
        weight_values = self._get_weight_values(
            layer_name='seq2seq_encoder_factored_lstm',
            weight_name='kernel_S_{}'.format(self.mode))

        # save kernel_S weights
        file_path = '{}.kernel_S.{}.npy'.format(path, self.mode)
        if (not os.path.exists(file_path)) or overwrite:
            print('save factored weight for emotion', self.mode)
            np.save(file_path, weight_values)

    def load(self, path):
        """Rewrite load model, only load weight from decoder and kernel_S encoder

        Arguments:
            path {str} -- string path for load model
        """
        initial_weight_kernel_S = self._get_weight_values(
            layer_name='seq2seq_encoder_factored_lstm',
            weight_name='kernel_S_{}'.format(self.mode))
        self.model_decoder.load_weights(path, by_name=True, skip_mismatch=True)

        try:
            kernel_S_value = np.load('{}.kernel_S.{}.npy'.format(
                path, self.mode))
        except IOError as e:
            print(e)
            print('But it\'s ok, it will be skipped')
            kernel_S_value = initial_weight_kernel_S

        self._set_weight_values(
            layer_name='seq2seq_encoder_factored_lstm',
            weight_values=kernel_S_value,
            weight_name='kernel_S_{}'.format(self.mode))

    def set_encoder_weights(self, stylenet):
        """set encoder weights from stylenet model decoder

        Arguments:
            stylenet {architecture} -- stylenet instance
        """
        assert self.injection_mode == stylenet.injection_mode
        assert self.transfer_values_size == stylenet.transfer_values_size
        assert self.encoder_lstm_layers == stylenet.lstm_layers
        assert self.state_size == stylenet.state_size
        assert self.factored_size == stylenet.factored_size
        assert self.embedding_size == stylenet.embedding_size
        assert self.num_words == stylenet.num_words

        initial_kernel_S_weight = self._get_weight_values(
            layer_name='seq2seq_encoder_factored_lstm',
            weight_name='kernel_S_{}'.format(self.mode))

        intersection_layers = zip(stylenet.model_decoder.layers,
                                  self.model_encoder.layers)
        for i, _ in enumerate(intersection_layers):
            w = stylenet.model_decoder.layers[i].get_weights()
            self.model_encoder.layers[i].set_weights(w)

        self._set_weight_values(
            layer_name='seq2seq_encoder_factored_lstm',
            weight_values=initial_kernel_S_weight,
            weight_name='kernel_S_{}'.format(self.mode))

    def predict(self,
                transfer_values,
                input_tokens,
                token_start,
                token_end,
                k=3,
                max_tokens=30):

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        states = self.model_encoder.predict([[transfer_values], [input_tokens]])
        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        token_int = token_start
        outputs_tokens = [(0, [token_int])]
        count_tokens = 0

        while count_tokens < max_tokens:

            tmp = []
            is_end_token = True
            for output_tokens in outputs_tokens:
                token_int = output_tokens[1][-1]
                if token_int == token_end:
                    tmp.append(output_tokens)
                    continue

                is_end_token = False
                decoder_input_data[0, count_tokens] = token_int
                x_data = [decoder_input_data] + states

                decoder_output = self.model_decoder.predict(x_data)

                tokens_pred = decoder_output[0, count_tokens, :]
                tokens_pred = softmax(tokens_pred)
                tokens_int = tokens_pred.argsort()[-k:][::-1]

                for token_int in tokens_int:
                    score = output_tokens[0] + tokens_pred[token_int]
                    tokens = output_tokens[1] + [token_int]
                    tmp.append((score, tokens))

            if is_end_token:
                break

            outputs_tokens = sorted(tmp, key=lambda t: t[0])[-k:]

            count_tokens += 1

        return np.array(outputs_tokens[0][1])
