from keras.layers import Activation, LSTM, Recurrent
import keras
from keras.engine import InputSpec, Layer
from keras import backend as K
from keras import initializations, activations, regularizers
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import Regularizer

def default_parameters():
    parameters = {}
    parameters['top_word_num'] = 5000
    parameters['word_dim'] = 60
    parameters['hidden_dim1'] = 90
    # Dimension of the chat representation
    parameters['hidden_dim2'] = 150
    # Dense hidden neuron number:
    parameters['dense_hidden'] = 100
    # Droupout
    parameters['dropout'] = 0.0
    # Size of the batch
    parameters['batch_size'] = 256
    #max epoch number
    parameters['max_epoch_number'] = 150
    #Validation data size
    parameters['validation_size'] = 0.1

    # Use Chat-LSTM
    parameters['use_chat_LSTM'] = True
    # Use direction regularization
    parameters['direction_reg_on'] = False
    #Use
    parameters['direction_reg'] = 0.1

    # Bucket size:
    parameters['message_length'] = 40
    parameters['chat_length'] = 40

    return parameters


class ChatLSTM(LSTM):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = int(input_shape[2] / 2)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 4 all-zero tensors of shape (output_dim)

            self.states = [None, None, None, None]

        self.W_i = self.init((self.input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((self.input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((self.input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((self.input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        self.W_s = self.init((self.output_dim, self.output_dim),
                             name='{}_W_s'.format(self.name))
        self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.W_e = self.init((self.output_dim, self.output_dim),
                             name='{}_W_e'.format(self.name))
        self.U_e = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_e'.format(self.name))
        self.b_e = K.zeros((self.output_dim,), name='{}_b_e'.format(self.name))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o,
                                  self.W_s, self.U_s, self.b_s,
                                  self.W_e, self.U_e, self.b_e]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[3],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        x1 = x[:, :self.input_dim]
        x2 = x[:, self.input_dim:]
        h1_tm1 = states[0]
        c1_tm1 = states[1]
        h2_tm1 = states[2]
        c2_tm1 = states[3]
        B_U = states[4]
        B_W = states[5]
        x1_i = K.dot(x1 * B_W[0], self.W_i) + self.b_i
        x1_f = K.dot(x1 * B_W[1], self.W_f) + self.b_f
        x1_c = K.dot(x1 * B_W[2], self.W_c) + self.b_c
        x1_o = K.dot(x1 * B_W[3], self.W_o) + self.b_o
        x1_s = K.dot(h2_tm1 * B_U[6], self.W_s) + self.b_s
        x1_e = K.dot(h2_tm1 * B_U[7], self.W_e) + self.b_e

        x2_i = K.dot(x2 * B_W[0], self.W_i) + self.b_i
        x2_f = K.dot(x2 * B_W[1], self.W_f) + self.b_f
        x2_c = K.dot(x2 * B_W[2], self.W_c) + self.b_c
        x2_o = K.dot(x2 * B_W[3], self.W_o) + self.b_o
        x2_s = K.dot(h1_tm1 * B_U[6], self.W_s) + self.b_s
        x2_e = K.dot(h1_tm1 * B_U[7], self.W_e) + self.b_e

        i1 = self.inner_activation(x1_i + K.dot(h1_tm1 * B_U[0], self.U_i))
        i2 = self.inner_activation(x2_i + K.dot(h2_tm1 * B_U[0], self.U_i))
        f1 = self.inner_activation(x1_f + K.dot(h1_tm1 * B_U[1], self.U_f))
        f2 = self.inner_activation(x2_f + K.dot(h2_tm1 * B_U[1], self.U_f))

        s1 = self.inner_activation(x1_s + K.dot(h1_tm1 * B_U[4], self.U_s))
        s2 = self.inner_activation(x2_s + K.dot(h2_tm1 * B_U[4], self.U_s))
        e1 = self.activation(x1_e + K.dot(h1_tm1 * B_U[5], self.U_e))
        e2 = self.activation(x2_e + K.dot(h2_tm1 * B_U[5], self.U_e))

        c1 = f1 * c1_tm1 + i1 * self.activation(x1_c + K.dot(h1_tm1 * B_U[2], self.U_c)) + s1 * e1
        c2 = f2 * c2_tm1 + i2 * self.activation(x2_c + K.dot(h2_tm1 * B_U[2], self.U_c)) + s2 * e2
        o1 = self.inner_activation(x1_o + K.dot(h1_tm1 * B_U[3], self.U_o))
        o2 = self.inner_activation(x2_o + K.dot(h2_tm1 * B_U[3], self.U_o))

        h1 = o1 * self.activation(c1)
        h2 = o2 * self.activation(c2)
        return K.concatenate([h1, h2]), [h1, h2, c1, c2]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(8)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(8)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim * 2)


class DirectionRegularizer(Regularizer):
    def __init__(self, batch_size=1, alfa=0.):
        self.alfa = K.cast_to_floatx(alfa)
        self.batch_size = batch_size

    def __call__(self, x):
        dim = int(K.int_shape(x)[-1] / 2)
        v1 = x[:, :dim]
        v1 = v1 * (1 / K.sqrt(K.sum(K.square(v1))))
        v2 = x[:, dim:]
        v2 = v2 * (1 / K.sqrt(K.sum(K.square(v2))))
        regularization = self.alfa * (self.batch_size - K.sum(K.sum(K.batch_dot(v1, v2, axes=[1, 1]), axis=1)))
        return regularization / self.batch_size

    def get_config(self):
        return {'name': self.__class__.__name__,
                'alfa': float(self.alfa)}


class ActivityDirectionRegularizer(Layer):
    def __init__(self, batch_size=1, alfa=0., **kwargs):
        self.supports_masking = True
        self.alfa = alfa
        self.batch_size = batch_size
        super(ActivityDirectionRegularizer, self).__init__(**kwargs)
        self.activity_regularizer = DirectionRegularizer(self.batch_size, self.alfa)
        self.regularizers = [self.activity_regularizer]

    def get_config(self):
        config = {'alfa': self.alfa}
        base_config = super(ActivityDirectionRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))