from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.recurrent import LSTMCell


class FactoredLSTMCell(LSTMCell):

    def __init__(self, units, modes, **kwargs):
        super(FactoredLSTMCell, self).__init__(units, **kwargs)
        self.modes = modes

    def build(self, input_shape):
        super(FactoredLSTMCell, self).build(input_shape)
        input_dim = input_shape[-1]

        self.kernels = {}
        for mode in self.modes:
            self.kernels[mode] = self.add_weight(
                shape=(input_dim, self.units * 4),
                name='kernel_{}'.format(mode),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

    def call(self, inputs, states, mode='factual', training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous cell state

        if mode not in self.modes:
            raise ValueError('{} not valid, use {} instead'.format(
                mode, self.modes))

        W_x = self.kernels[mode]
        W_h = self.recurrent_kernel

        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        x_i = K.dot(inputs_i, W_x[:, :self.units])
        x_f = K.dot(inputs_f, W_x[:, self.units:self.units * 2])
        x_c = K.dot(inputs_c, W_x[:, self.units * 2:self.units * 3])
        x_o = K.dot(inputs_o, W_x[:, self.units * 3:])

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


class FactoredLSTM():

    def __init__(self, **kwargs):
        pass
