from keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Input, Dense, LSTM, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import np
from nn import FactoredLSTM
from loss import sparse_cross_entropy
import tensorflow as tf


class NIC():

    def __init__(self,
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128):
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.model = None
        self._build()

    def _build(self):
        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        decoder_transfer_map = Dense(
            self.state_size, activation='tanh', name='decoder_transfer_map')

        # word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding')

        # decoder LSTM
        decoder_lstm = LSTM(
            self.state_size, name='decoder_lstm', return_sequences=True)
        decoder_dense = Dense(
            self.num_words, activation='linear', name='decoder_output')

        # connect decoder
        initial_state = decoder_transfer_map(transfer_values_input)
        net = decoder_input
        net = decoder_embedding(net)
        net = decoder_lstm(net, initial_state=(initial_state, initial_state))
        decoder_output = decoder_dense(net)

        # create model
        self.model = Model(
            inputs=[transfer_values_input, decoder_input],
            outputs=[decoder_output])

        # RMS optimizer
        optimizer = RMSprop(lr=1e-3)

        # compile model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])


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
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128,
                 factored_size=256):
        self.mode = mode
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.factored_size = factored_size
        self.model = None
        self._build()

    def _build(self):
        # image embedding
        self.transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        self.decoder_transfer_map = Dense(
            self.state_size,
            activation='tanh',
            name='decoder_transfer_map',
            trainable=self.mode == 'factual')

        # word embedding
        self.decoder_input = Input(shape=(None,), name='decoder_input')
        self.decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding',
            trainable=self.mode == 'factual')

        # decoder LSTM
        self.decoder_factored_lstm = FactoredLSTM(
            self.state_size,
            mode=self.mode,
            name='decoder_factored_lstm',
            return_sequences=True)
        self.decoder_dense = Dense(
            self.num_words,
            activation='linear',
            name='decoder_output',
            trainable=self.mode == 'factual')

        # connect decoder
        initial_state = self.decoder_transfer_map(self.transfer_values_input)
        net = self.decoder_input
        net = self.decoder_embedding(net)
        net = self.decoder_factored_lstm(
            net, initial_state=(initial_state, initial_state))
        decoder_output = self.decoder_dense(net)

        # create model
        self.model = Model(
            inputs=[self.transfer_values_input, self.decoder_input],
            outputs=[decoder_output])

        # RMS optimizer
        optimizer = RMSprop(lr=1e-3)

        # compile model
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])

    def save(self, path):
        self.model.save_weights(path, overwrite=True)
        for layer in self.model.layers:
            if layer.name == 'decoder_factored_lstm':
                for weight, value in zip(layer.weights, layer.get_weights()):
                    name = weight.name.split(':')[0].split('/')[1]
                    if name == 'kernel_S_{}'.format(self.mode):
                        np.save('{}.kernel_S.{}'.format(path, self.mode), value)
                        break
                break

    def load(self, path):
        self.model.load_weights(path, by_name=True)
        kernel_S_value = np.load('{}.kernel_S.{}'.format(path, self.mode))
        for i, layer in enumerate(self.model.layers):
            if layer.name == 'decoder_factored_lstm':
                weights = []
                for weight, value in zip(layer.weights, layer.get_weights()):
                    name = weight.name.split(':')[0].split('/')[1]
                    if name == 'kernel_S_{}'.format(self.mode):
                        weights.append(kernel_S_value)
                    else:
                        weights.append(value)
                self.model.layers[i].set_weights(weights)
                break
