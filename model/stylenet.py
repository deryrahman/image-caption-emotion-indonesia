from keras.layers import Input, Dense, Embedding, Concatenate, RepeatVector, Lambda
from keras.applications.resnet_v2 import ResNet152V2
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import np
# from keras.utils import plot_model
from nn import FactoredLSTM
from loss import sparse_cross_entropy
import tensorflow as tf
import os


class StyleNet():

    def __init__(self,
                 mode='factual',
                 include_transfer_value=True,
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
        self.model_encoder = None
        self.model_decoder = None
        self.model_decoder_partial = None
        self._build()

    def _build(self):
        # encoder ResNet
        image_model = ResNet152V2(include_top=True, weights='imagenet')
        for layer in image_model.layers:
            layer.trainable = False
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

        # decoder LSTM
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

        def connect_lstm(states, uniform_state, decoder_net):

            for i in range(self.lstm_layers):
                decoder_net, state_h, state_c = decoder_factored_lstm[i](
                    decoder_net, initial_state=states)

                if not uniform_state:
                    states = [state_h, state_c]

            return decoder_net

        def connect_decoder(encoder_output, decoder_input):

            decoder_net = decoder_embedding(decoder_input)

            if encoder_output is None:
                states = None
                decoder_net = connect_lstm(states, False, decoder_net)
                decoder_output = decoder_dense(decoder_net)
                return decoder_output

            decoder_transfer = decoder_transfer_map(encoder_output)

            if self.injection_mode == 'init':
                states = [decoder_transfer, decoder_transfer]
                decoder_net = connect_lstm(states, True, decoder_net)

            if self.injection_mode == 'pre':
                states = None
                decoder_init = RepeatVector(1)(decoder_transfer)
                decoder_net = Concatenate(axis=1)([decoder_init, decoder_net])
                decoder_net = connect_lstm(states, False, decoder_net)
                # shift output lstm 1 step to the right
                decoder_net = Lambda(lambda x: x[:, 1:, :])(decoder_net)

            decoder_output = decoder_dense(decoder_net)

            return decoder_output

        # connect full model
        encoder_output = transfer_layer.output
        decoder_output = connect_decoder(encoder_output, decoder_input)
        self.model = Model(
            inputs=[image_model.input, decoder_input], outputs=[decoder_output])

        # connect encoder ResNet
        self.model_encoder = Model(
            inputs=[image_model.input], outputs=[transfer_layer.output])

        # connect decoder FactoredLSTM
        decoder_output = connect_decoder(transfer_values_input, decoder_input)
        self.model_decoder = Model(
            inputs=[transfer_values_input, decoder_input],
            outputs=[decoder_output])

        # connect decoder FactoredLSTM without transfer value
        decoder_output = connect_decoder(None, decoder_input)
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

    def save(self, path, overwrite):
        """Rewrite save model, only save weight from decoder
 
        Arguments:
            path {str} -- string path to save model
            overwrite {bool} -- overwrite existing model or not
        """
        if self.trainable_model:
            self.model_decoder.save_weights(path, overwrite=overwrite)

        # weights values for kernel_S
        weight_values = []
        for layer in self.model.layers:
            if layer.name[:-2] != 'decoder_factored_lstm':
                continue
            for weight, value in zip(layer.weights, layer.get_weights()):
                name = weight.name.split(':')[0].split('/')[1]
                if name != 'kernel_S_{}'.format(self.mode):
                    continue
                weight_values.append([value])
                break

        # save kernel_S weights
        file_path = '{}.kernel_S.{}.npy'.format(path, self.mode)
        if (not os.path.exists(file_path)) or overwrite:
            print('save factored weight for emotion', self.mode)
            np.save(file_path, weight_values)

    def load(self, path):
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

    def _get_weight_values(self, layer_name, weight_name=None):
        weight_values = []
        for i, layer in enumerate(self.model_decoder.layers):
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
        for layer in self.model_decoder.layers:
            if layer.name[:-2] == layer_name:
                layer_target_i += 1
        if layer_target_i != len(weight_values):
            raise ValueError(
                'length of weight_values didn\'t match with layer count in model, expect {} got {}'
                .format(layer_target_i, len(weight_values)))
        layer_target_i = 0
        for i, layer in enumerate(self.model_decoder.layers):
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
                self.model_decoder.layers[i].set_weights(weights)
                layer_target_i += 1
