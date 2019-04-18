from keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import np
from nn import FactoredLSTM
from loss import sparse_cross_entropy
from keras.utils import plot_model
import tensorflow as tf
import os


class NIC():

    def __init__(self,
                 include_transfer_value=False,
                 injection_mode='init',
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128,
                 learning_rate=0.0002,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 lstm_layers=1):
        self.injection_mode = injection_mode
        self.include_transfer_value = include_transfer_value
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lstm_layers = lstm_layers
        self.model = None
        self._build()

    def _build(self):
        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        if self.injection_mode == 'init':
            decoder_units = self.state_size
        elif self.injection_mode == 'pre':
            decoder_units = self.embedding_size
        decoder_transfer_map = Dense(
            decoder_units, activation='tanh', name='decoder_transfer_map')
        decoder_transfer_map_transform = RepeatVector(1)
        concatenate = Concatenate(axis=1)

        # word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding')

        # decoder LSTM
        decoder_lstm = []
        for i in range(self.lstm_layers):
            decoder_lstm.append(
                LSTM(
                    self.state_size,
                    name='decoder_lstm_{}'.format(i),
                    return_sequences=True))
        decoder_dense = Dense(
            self.num_words, activation='linear', name='decoder_output')
        decoder_step = Lambda(lambda x: x[:, 1:, :])

        # connect decoder
        net = decoder_input
        net = decoder_embedding(net)
        if self.include_transfer_value:
            initial_state = decoder_transfer_map(transfer_values_input)
            if self.injection_mode == 'init':
                for i in range(self.lstm_layers):
                    net = decoder_lstm[i](
                        net, initial_state=(initial_state, initial_state))
            elif self.injection_mode == 'pre':
                initial_state = decoder_transfer_map_transform(initial_state)
                net = concatenate([initial_state, net])
                for i in range(self.lstm_layers):
                    net = decoder_lstm[i](net)
                net = decoder_step(net)
        else:
            for i in range(self.lstm_layers):
                net = decoder_lstm[i](net)
        decoder_output = decoder_dense(net)

        # create model
        self.model = Model(
            inputs=[transfer_values_input, decoder_input]
            if self.include_transfer_value else [decoder_input],
            outputs=[decoder_output])

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        # compile model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])
        plot_model(self.model, to_file='nic.png', show_shapes=True)


class EncoderResNet152():

    def __init__(self, weights='imagenet'):
        self.weights = weights
        self.model = None
        self._build()

    def _build(self):
        # from pretrained model
        image_model = ResNet152V2(include_top=True, weights=self.weights)
        transfer_layer = image_model.get_layer('avg_pool')
        self.model = Model(
            inputs=image_model.input, outputs=transfer_layer.output)


class StyleNet():

    def __init__(self,
                 mode='factual',
                 include_transfer_value=True,
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
        self.include_transfer_value = include_transfer_value
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
        self._build()

    def _build(self):
        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        if self.injection_mode == 'init':
            decoder_units = self.state_size
        elif self.injection_mode == 'pre':
            decoder_units = self.embedding_size
        decoder_transfer_map = Dense(
            decoder_units,
            activation='tanh',
            name='decoder_transfer_map',
            trainable=self.mode == 'factual')
        decoder_transfer_map_transform = RepeatVector(1)
        concatenate = Concatenate(axis=1)

        # word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding',
            trainable=self.mode == 'factual')

        # decoder LSTM
        decoder_factored_lstm = []
        for i in range(self.lstm_layers):
            decoder_factored_lstm.append(
                FactoredLSTM(
                    self.state_size,
                    mode=self.mode,
                    name='decoder_factored_lstm_{}'.format(i),
                    return_sequences=True))
        decoder_dense = Dense(
            self.num_words,
            activation='linear',
            name='decoder_output',
            trainable=self.mode == 'factual')
        decoder_step = Lambda(lambda x: x[:, 1:, :])

        # connect decoder
        net = decoder_input
        net = decoder_embedding(net)
        if self.include_transfer_value:
            initial_state = decoder_transfer_map(transfer_values_input)
            if self.injection_mode == 'init':
                for i in range(self.lstm_layers):
                    net = decoder_factored_lstm[i](
                        net, initial_state=(initial_state, initial_state))
            elif self.injection_mode == 'pre':
                initial_state = decoder_transfer_map_transform(initial_state)
                net = concatenate([initial_state, net])
                for i in range(self.lstm_layers):
                    net = decoder_factored_lstm[i](net)
                net = decoder_step(net)
        else:
            for i in range(self.lstm_layers):
                net = decoder_factored_lstm[i](net)
        decoder_output = decoder_dense(net)

        # create model
        self.model = Model(
            inputs=[transfer_values_input, decoder_input]
            if self.include_transfer_value else [decoder_input],
            outputs=[decoder_output])

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        # compile model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])
        plot_model(self.model, to_file='stylenet.png', show_shapes=True)

    def save(self, path, overwrite):
        if self.mode == 'factual':
            self.model.save_weights(path, overwrite=overwrite)
        weight_values = []
        for layer in self.model.layers:
            if layer.name[:-2] == 'decoder_factored_lstm':
                for weight, value in zip(layer.weights, layer.get_weights()):
                    name = weight.name.split(':')[0].split('/')[1]
                    if name == 'kernel_S_{}'.format(self.mode):
                        weight_values.append([value])
                        break
        file_path = '{}.kernel_S.{}.npy'.format(path, self.mode)
        if (not os.path.exists(file_path)) or overwrite:
            print('save factored weight for emotion', self.mode)
            np.save(file_path, weight_values)

    def load(self, path):
        initial_weight_kernel_S = self._get_weight_values(
            layer_name='decoder_factored_lstm',
            weight_name='kernel_S_{}'.format(self.mode))
        self.model.load_weights(path, by_name=True, skip_mismatch=True)
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

    def _get_weight_values(self, layer_name, weight_name=None):
        weight_values = []
        for i, layer in enumerate(self.model.layers):
            if layer.name[:-2] == layer_name:
                if not weight_name:
                    weight_values.append(layer.get_weights())
                    continue
                for weight, value in zip(layer.weights, layer.get_weights()):
                    name = weight.name.split(':')[0].split('/')[1]
                    if name != weight_name:
                        continue
                    weight_values.append([value])
                    break
        return weight_values

    def _set_weight_values(self, layer_name, weight_values, weight_name=None):
        layer_target_i = 0
        for layer in self.model.layers:
            if layer.name[:-2] == layer_name:
                layer_target_i += 1
        if layer_target_i != len(weight_values):
            raise ValueError(
                'length of weight_values didn\'t match with layer count in model, expect {} got {}'
                .format(layer_target_i, len(weight_values)))
        layer_target_i = 0
        for i, layer in enumerate(self.model.layers):
            if layer.name[:-2] == layer_name:
                weights = []
                if not weight_name:
                    weights = weight_values[layer_target_i]
                else:
                    for weight, value in zip(layer.weights,
                                             layer.get_weights()):
                        name = weight.name.split(':')[0].split('/')[1]
                        if name != weight_name:
                            weights.append(value)
                            continue
                        weights.append(weight_values[layer_target_i][0])
                self.model.layers[i].set_weights(weights)
                layer_target_i += 1
