from keras.layers import Input, Dense, Embedding, Concatenate, RepeatVector, Lambda
from keras.applications.resnet_v2 import ResNet152V2
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import np
# from keras.utils import plot_model
from nn import FactoredLSTM
from loss import sparse_cross_entropy
from model.base import RichModel
import tensorflow as tf
import os


class StyleNet(RichModel):

    def __init__(self,
                 mode='factual',
                 trainable_model=True,
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
                 lstm_layers=1):
        self.injection_mode = injection_mode
        self.trainable_model = trainable_model
        self.mode = mode
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.factored_size = factored_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lstm_layers = lstm_layers
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self.model_decoder_partial = None
        self._build()

    def _build(self):
        # encoder ResNet
        image_model = ResNet152V2(include_top=True, weights='imagenet')
        transfer_layer = image_model.get_layer('avg_pool')

        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        decoder_units = {
            'init': self.state_size,
            'pre': self.embedding_size
        }.get(self.injection_mode, self.state_size)
        decoder_transfer_map = Dense(
            decoder_units,
            activation='tanh',
            name='decoder_transfer_map',
            trainable=self.trainable_model)

        # word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding',
            trainable=self.trainable_model)

        # decoder Factored LSTM
        decoder_factored_lstm = [
            FactoredLSTM(
                self.state_size,
                mode=self.mode,
                trainable_model=self.trainable_model,
                factored_dim=self.factored_size,
                name='decoder_factored_lstm_{}'.format(i),
                return_state=True,
                return_sequences=True) for i in range(self.lstm_layers)
        ]
        decoder_dense = Dense(
            self.num_words,
            activation='linear',
            name='decoder_output',
            trainable=self.trainable_model)

        def connect_lstm(states, uniform_state, lstm_layers, net):

            for i in range(len(lstm_layers)):
                net, state_h, state_c = lstm_layers[i](
                    net, initial_state=states)

                if not uniform_state:
                    states = [state_h, state_c]

            return net, state_h, state_c

        def connect_decoder(encoder_net, decoder_input):

            decoder_net = decoder_embedding(decoder_input)

            if encoder_net is None:
                states = None
                decoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=False,
                    lstm_layers=decoder_factored_lstm,
                    net=decoder_net)

                return decoder_net, state_h, state_c

            decoder_transfer = decoder_transfer_map(encoder_net)

            if self.injection_mode == 'init':
                states = [decoder_transfer, decoder_transfer]
                decoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=True,
                    lstm_layers=decoder_factored_lstm,
                    net=decoder_net)

                return decoder_net, state_h, state_c

            if self.injection_mode == 'pre':
                states = None
                decoder_init = RepeatVector(1)(decoder_transfer)
                decoder_net = Concatenate(axis=1)([decoder_init, decoder_net])
                decoder_net, state_h, state_c = connect_lstm(
                    states=states,
                    uniform_state=False,
                    lstm_layers=decoder_factored_lstm,
                    net=decoder_net)
                # shift output lstm 1 step to the right
                decoder_net = Lambda(lambda x: x[:, 1:, :])(decoder_net)

                return decoder_net, state_h, state_c

            return None, None, None

        # connect full model
        encoder_net = transfer_layer.output
        decoder_net, _, _ = connect_decoder(encoder_net, decoder_input)
        decoder_output = decoder_dense(decoder_net)
        self.model = Model(
            inputs=[image_model.input, decoder_input], outputs=[decoder_output])

        # connect encoder ResNet
        self.model_encoder = Model(
            inputs=[image_model.input], outputs=[transfer_layer.output])

        # connect decoder FactoredLSTM
        decoder_net, _, _ = connect_decoder(transfer_values_input,
                                            decoder_input)
        decoder_output = decoder_dense(decoder_net)
        self.model_decoder = Model(
            inputs=[transfer_values_input, decoder_input],
            outputs=[decoder_output])

        # connect decoder FactoredLSTM without transfer value
        decoder_net, _, _ = connect_decoder(None, decoder_input)
        decoder_output = decoder_dense(decoder_net)
        self.model_decoder_partial = Model(
            inputs=[decoder_input], outputs=[decoder_output])

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        # compile model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model_decoder.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])
        self.model_decoder_partial.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])
        # plot_model(self.model, to_file='stylenet.png', show_shapes=True)

    def predict(self, image, token_start, token_end, k=3, max_tokens=30):

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        image_batch = np.expand_dims(image, axis=0)
        transfer_values = self.model_encoder.predict(image_batch)

        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        token_int = token_start
        outputs_tokens = [(0, [token_int])]
        count_tokens = 0
        tmp = []

        while tmp != outputs_tokens and count_tokens < max_tokens:

            tmp = []
            for output_tokens in outputs_tokens:
                token_int = output_tokens[1][-1]
                if token_int == token_end:
                    tmp.append(output_tokens)
                    continue

                decoder_input_data[0, count_tokens] = token_int
                x_data = [transfer_values, decoder_input_data]

                decoder_output = self.model_decoder.predict(x_data)

                tokens_pred = decoder_output[0, count_tokens, :]
                tokens_pred = softmax(tokens_pred)
                tokens_int = tokens_pred.argsort()[-k:][::-1]

                for token_int in tokens_int:
                    score = output_tokens[0] + tokens_pred[token_int]
                    tokens = output_tokens[1] + [token_int]
                    tmp.append((score, tokens))

            outputs_tokens = sorted(tmp, key=lambda t: t[0])[-k:]

            count_tokens += 1

        return np.array(outputs_tokens[0][1]), np.array(transfer_values[0])

    def save(self, path, overwrite):
        """Rewrite save model, only save weight from decoder

        Arguments:
            path {str} -- string path to save model
            overwrite {bool} -- overwrite existing model or not
        """
        if self.trainable_model:
            self.model_decoder.save_weights(path, overwrite=overwrite)

        # weights values for kernel_S
        weight_values = self._get_weight_values(
            layer_name='decoder_factored_lstm',
            weight_name='kernel_S_{}'.format(self.mode))

        # save kernel_S weights
        file_path = '{}.kernel_S.{}.npy'.format(path, self.mode)
        if (not os.path.exists(file_path)) or overwrite:
            print('save factored weight for emotion', self.mode)
            np.save(file_path, weight_values)

    def load(self, path):
        """Rewrite load model, only load weight from decoder

        Arguments:
            path {str} -- string path for load model
        """
        initial_weight_kernel_S = self._get_weight_values(
            layer_name='decoder_factored_lstm',
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
            layer_name='decoder_factored_lstm',
            weight_values=kernel_S_value,
            weight_name='kernel_S_{}'.format(self.mode))
