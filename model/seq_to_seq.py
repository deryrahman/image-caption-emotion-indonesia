from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import np
from loss import sparse_cross_entropy
from nn import FactoredLSTM
import tensorflow as tf


class Seq2Seq():

    def __init__(self,
                 mode,
                 trainable_factor=True,
                 num_words=10000,
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
        self.trainable_factor = trainable_factor
        self.num_words = num_words
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
        self.encoder_model = None
        self._build()

    def _build(self):
        # encoder word embedding
        encoder_input = Input(shape=(None,), name='encoder_input')
        encoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='encoder_embedding',
            trainable=self.trainable_factor)

        # encoder LSTM
        encoder_factored_lstm = []
        for i in range(self.encoder_lstm_layers):
            encoder_factored_lstm.append(
                FactoredLSTM(
                    self.state_size,
                    mode=self.mode,
                    trainable_factor=self.trainable_factor,
                    factored_dim=self.factored_size,
                    name='encoder_factored_lstm_{}'.format(i),
                    return_state=True))

        # decoder word embedding
        decoder_input = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(
            input_dim=self.num_words,
            output_dim=self.embedding_size,
            name='decoder_embedding')

        # decoder LSTM
        decoder_lstm = []
        for i in range(self.decoder_lstm_layers):
            decoder_lstm.append(
                LSTM(
                    self.state_size,
                    name='decoder_lstm_{}'.format(i),
                    return_sequences=True,
                    return_state=True))
        decoder_dense = Dense(
            self.num_words, activation='linear', name='decoder_output')

        encoder_net = encoder_embedding(encoder_input)
        encoder_states = []
        for i in range(len(encoder_factored_lstm)):
            if len(encoder_states) == 0:
                encoder_net, encoder_state_h, encoder_state_c = encoder_factored_lstm[
                    i](encoder_net)
            else:
                encoder_net, encoder_state_h, encoder_state_c = encoder_factored_lstm[
                    i](encoder_net, initial_state=encoder_states)
            encoder_states = [encoder_state_h, encoder_state_c]

        decoder_net = decoder_embedding(decoder_input)
        for i in range(len(decoder_lstm)):
            decoder_net, _, _ = decoder_lstm[i](
                decoder_net, initial_state=encoder_states)

        decoder_output = decoder_dense(decoder_net)

        # create model
        self.model = Model(
            inputs=[encoder_input, decoder_input], outputs=[decoder_output])

        # RMSprop optimizer
        optimizer = RMSprop(lr=1e-3)

        # model compile
        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        self.model.compile(
            optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=decoder_target)

        # encoder model
        self.encoder_model = Model(
            inputs=[encoder_input], outputs=encoder_states)

        decoder_state_input_h = Input(shape=(self.state_size,))
        decoder_state_input_c = Input(shape=(self.state_size,))
        decoder_states_input = [decoder_state_input_h, decoder_state_input_c]

        # decoder model
        decoder_net = decoder_embedding(decoder_input)
        for i in range(len(decoder_lstm)):
            decoder_net, _, _ = decoder_lstm[i](
                decoder_net, initial_state=decoder_states_input)
        decoder_output = decoder_dense(decoder_net)

        self.decoder_model = Model(
            inputs=[decoder_input] + decoder_states_input,
            outputs=[decoder_output])

    def save(self, path, overwrite):
        print('save hold')

    def load(self, path):
        print('load hold')

    def predict(self, input_tokens, token_start, token_end, max_tokens=30):
        states_value = self.encoder_model.predict(input_tokens)
        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        token_int = token_start
        output_tokens = [token_int]
        count_tokens = 0

        while token_int != token_end and count_tokens < max_tokens:
            decoder_input_data[0, count_tokens] = token_int
            x_data = [decoder_input_data] + states_value
            decoder_output = self.decoder_model.predict(x_data)

            token_onehot = decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_onehot)

            output_tokens.append(token_int)

            count_tokens += 1

        return output_tokens