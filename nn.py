from keras import backend as K, regularizers, initializers
from keras.utils.np_utils import np
from keras.layers.recurrent import LSTMCell, LSTM


class FactoredLSTMCell(LSTMCell):

    def __init__(self, units, factored_dim, mode, **kwargs):
        self.factored_dim = factored_dim
        self.mode = mode
        super(FactoredLSTMCell, self).__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if type(self.recurrent_initializer).__name__ == 'Identity':

            def recurrent_identity(shape, gain=1.):
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        def bias_initializer(_, *args, **kwargs):
            return K.concatenate([
                self.bias_initializer((self.units,), *args, **kwargs),
                initializers.Ones()((self.units,), *args, **kwargs),
                self.bias_initializer((self.units * 2,), *args, **kwargs),
            ])

        self.kernel_S = self.add_weight(
            shape=(self.factored_dim, self.factored_dim * 4),
            name='kernel_S_{}'.format(self.mode),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.kernel_V = self.add_weight(
            shape=(input_dim, self.factored_dim * 4),
            name='kernel_V',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.mode == 'factual')
        self.kernel_U = self.add_weight(
            shape=(self.factored_dim, self.units * 4),
            name='kernel_U',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.mode == 'factual')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=self.mode == 'factual')
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer=bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=self.mode == 'factual')
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous cell state

        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        V = self.kernel_V
        x_i = K.dot(inputs_i, V[:, :self.factored_dim])
        x_f = K.dot(inputs_f, V[:, self.factored_dim:self.factored_dim * 2])
        x_c = K.dot(inputs_c, V[:, self.factored_dim * 2:self.factored_dim * 3])
        x_o = K.dot(inputs_o, V[:, self.factored_dim * 3:])

        S = self.kernel_S
        x_i = K.dot(x_i, S[:, :self.factored_dim])
        x_f = K.dot(x_f, S[:, self.factored_dim:self.factored_dim * 2])
        x_c = K.dot(x_c, S[:, self.factored_dim * 2:self.factored_dim * 3])
        x_o = K.dot(x_o, S[:, self.factored_dim * 3:])

        U = self.kernel_U
        x_i = K.dot(x_i, U[:, :self.units])
        x_f = K.dot(x_f, U[:, self.units:self.units * 2])
        x_c = K.dot(x_c, U[:, self.units * 2:self.units * 3])
        x_o = K.dot(x_o, U[:, self.units * 3:])

        x_i = K.bias_add(x_i, self.bias[:self.units])
        x_f = K.bias_add(x_f, self.bias[self.units:self.units * 2])
        x_c = K.bias_add(x_c, self.bias[self.units * 2:self.units * 3])
        x_o = K.bias_add(x_o, self.bias[self.units * 3:])

        W_h = self.recurrent_kernel
        h_i = K.dot(h_tm1_i, W_h[:, :self.units])
        h_f = K.dot(h_tm1_f, W_h[:, self.units:self.units * 2])
        h_c = K.dot(h_tm1_c, W_h[:, self.units * 2:self.units * 3])
        h_o = K.dot(h_tm1_o, W_h[:, self.units * 3:])

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = self.activation(x_c + h_c)
        c = f * c_tm1 + i * c
        o = self.recurrent_activation(x_o + h_o)

        h = o * c

        return h, [h, c]


class FactoredLSTM(LSTM):

    def __init__(self,
                 units,
                 mode='factual',
                 factored_dim=256,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = FactoredLSTMCell(
            units,
            factored_dim=factored_dim,
            mode=mode,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation)
        super(LSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
