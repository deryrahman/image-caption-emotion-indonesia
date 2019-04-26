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
                 with_transfer_value=True,
                 trainable_model=True,
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128,
                 factored_size=256,
                 learning_rate=0.0002,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 lstm_layers=1,
                 dropout=0.0):
        self.trainable_model = trainable_model
        self.with_transfer_value = with_transfer_value
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
        self.dropout = dropout
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self._build()

    def _build(self):
        # encoder ResNet
        image_model = ResNet152V2(include_top=True, weights='imagenet')
        transfer_layer = image_model.get_layer('avg_pool')
        for layer in image_model.layers:
            layer.trainable = False

        # image embedding
        transfer_values_input = Input(
            shape=(self.transfer_values_size,), name='transfer_values_input')
        decoder_transfer_map = Dense(
            self.embedding_size,
            activation='tanh',
            name='decoder_transfer_map',
            trainable=self.trainable_model)
        decoder_transfer_map_transform = RepeatVector(
            1,
            name='decoder_transfer_map_transform',
            trainable=self.trainable_model)
        concatenate = Concatenate(
            axis=1, name='decoder_concatenate', trainable=self.trainable_model)

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
                return_sequences=True,
                recurrent_dropout=self.dropout,
                dropout=self.dropout) for i in range(self.lstm_layers)
        ]
        decoder_dense = Dense(
            self.num_words,
            activation='linear',
            name='decoder_output',
            trainable=self.trainable_model)
        decoder_step = Lambda(
            lambda x: x[:, 1:, :],
            name='decoder_step',
            trainable=self.trainable_model)

        def connect_lstm(lstm_layers, net):
            for i in range(len(lstm_layers)):
                net = lstm_layers[i](net)
            return net

        def connect_decoder(encoder_output, decoder_input):

            decoder_net = decoder_embedding(decoder_input)

            if encoder_output is None:
                decoder_net = connect_lstm(
                    lstm_layers=decoder_factored_lstm, net=decoder_net)
                return decoder_net

            decoder_transfer = decoder_transfer_map(encoder_output)
            decoder_transfer = decoder_transfer_map_transform(decoder_transfer)
            decoder_net = concatenate([decoder_transfer, decoder_net])
            decoder_net = connect_lstm(
                lstm_layers=decoder_factored_lstm, net=decoder_net)
            decoder_net = decoder_step(decoder_net)

            return decoder_net

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        if self.with_transfer_value:
            # connect decoder FactoredLSTM
            decoder_net = decoder_input
            encoder_output = transfer_values_input
            decoder_net = connect_decoder(encoder_output, decoder_net)
            decoder_output = decoder_dense(decoder_net)
            self.model_decoder = Model(
                inputs=[transfer_values_input, decoder_input],
                outputs=[decoder_output])

            decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
            self.model_decoder.compile(
                optimizer=optimizer,
                loss=sparse_cross_entropy,
                target_tensors=[decoder_target])
        else:
            # connect decoder FactoredLSTM without transfer value
            decoder_net = decoder_input
            decoder_net = connect_decoder(None, decoder_net)
            decoder_output = decoder_dense(decoder_net)
            self.model_decoder = Model(
                inputs=[decoder_input], outputs=[decoder_output])

            decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
            self.model_decoder.compile(
                optimizer=optimizer,
                loss=sparse_cross_entropy,
                target_tensors=[decoder_target])

        # connect encoder ResNet
        self.model_encoder = Model(
            inputs=[image_model.input], outputs=[transfer_layer.output])

        self.model = self.model_decoder
        # plot_model(self.model, to_file='stylenet.png', show_shapes=True)

    def predict(self, image, token_start, token_end, max_tokens=30):

        image_batch = np.expand_dims(image, axis=0)
        transfer_values = self.model_encoder.predict(image_batch)

        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        token_int = token_start
        output_tokens = [token_int]
        count_tokens = 0

        while token_int != token_end and count_tokens < max_tokens:
            decoder_input_data[0, count_tokens] = token_int
            x_data = [transfer_values, decoder_input_data]

            decoder_output = self.model_decoder.predict(x_data)

            token_onehot = decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_onehot)

            output_tokens.append(token_int)

            count_tokens += 1

        return np.array(output_tokens), np.array(transfer_values[0])

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
