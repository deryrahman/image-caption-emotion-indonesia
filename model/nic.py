from keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector, Lambda
from keras.models import Model
from keras.applications.resnet_v2 import ResNet152V2
from keras.optimizers import Adam
from loss import sparse_cross_entropy
from model.base import RichModel
# from keras.utils import plot_model
import tensorflow as tf


class NIC(RichModel):

    def __init__(self,
                 num_words=10000,
                 transfer_values_size=2048,
                 state_size=512,
                 embedding_size=128,
                 learning_rate=0.0002,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 lstm_layers=1,
                 dropout=0.5):
        self.num_words = num_words
        self.transfer_values_size = transfer_values_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self.model_decoder_partial = None
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
            1, name='decoder_transfer_map_transform')
        concatenate = Concatenate(axis=1, name='decoder_concatenate')

        # word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding',
            trainable=self.trainable_model)

        # decoder Factored LSTM
        decoder_lstm = [
            LSTM(
                self.state_size,
                name='decoder_lstm_{}'.format(i),
                return_sequences=True,
                recurrent_dropout=self.dropout,
                dropout=self.dropout) for i in range(self.lstm_layers)
        ]
        decoder_dense = Dense(
            self.num_words,
            activation='linear',
            name='decoder_dense',
            trainable=self.trainable_model)
        decoder_step = Lambda(lambda x: x[:, 1:, :], name='decoder_step')

        def connect_lstm(lstm_layers, net):
            for i in range(len(lstm_layers)):
                net = lstm_layers[i](net)
            return net

        def connect_decoder(encoder_output, decoder_input):

            decoder_net = decoder_embedding(decoder_input)

            if encoder_output is None:
                decoder_net = connect_lstm(
                    lstm_layers=decoder_lstm, net=decoder_net)
                return decoder_net

            decoder_transfer = decoder_transfer_map(encoder_output)
            decoder_transfer = decoder_transfer_map_transform(decoder_transfer)
            decoder_net = concatenate([decoder_transfer, decoder_net])
            decoder_net = connect_lstm(
                lstm_layers=decoder_lstm, net=decoder_net)
            decoder_net = decoder_step(decoder_net)

            return decoder_net

        # connect encoder ResNet
        self.model_encoder = Model(
            inputs=[image_model.input], outputs=[transfer_layer.output])

        # connect decoder LSTM
        decoder_net = decoder_input
        encoder_output = transfer_values_input
        decoder_net = connect_decoder(encoder_output, decoder_net)
        decoder_output = decoder_dense(decoder_net)
        self.model_decoder = Model(
            inputs=[transfer_values_input, decoder_input],
            outputs=[decoder_output])

        # connect decoder LSTM without transfer value
        decoder_net = decoder_input
        decoder_net = connect_decoder(None, decoder_net)
        decoder_output = decoder_dense(decoder_net)
        self.model_decoder_partial = Model(
            inputs=[decoder_input], outputs=[decoder_output])

        # Adam optimizer
        optimizer = Adam(
            lr=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon)

        # compile model decoder
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model_decoder.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])

        self.model_decoder_partial.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])

        self.model = self.model_decoder
        # plot_model(self.model, to_file='nic.png', show_shapes=True)

    def save(self, path, overwrite):
        self.model.save_weights(path, overwrite=overwrite)

    def load(self, path):
        self.model.load_weights(path, by_name=True, skip_mismatch=True)
