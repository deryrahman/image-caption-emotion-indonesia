from keras.layers import Input, Dense, LSTM, Embedding, Concatenate, RepeatVector, Lambda
from keras.models import Model
from keras.optimizers import Adam
from loss import sparse_cross_entropy
# from keras.utils import plot_model
import tensorflow as tf


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
        decoder_transfer_map_transform = RepeatVector(
            1, name='decoder_transfer_map_transform')
        concatenate = Concatenate(axis=1, name='decoder_concatenate')

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
        decoder_step = Lambda(lambda x: x[:, 1:, :], name='decoder_step')

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
        # plot_model(self.model, to_file='nic.png', show_shapes=True)

    def save(self, path, overwrite):
        self.model.save_weights(path, overwrite=overwrite)

    def load(self, path):
        self.model.load_weights(path, by_name=True, skip_mismatch=True)