import tensorflow as tf
import numpy as np
from utils import lognormdens, weight_variable, bias_variable



class MLP(object):

    def __init__(self, dims, activations, stddev=1., bias_value=0.0):
        self.dims = dims
        self.activations = activations
        self.layers = []
        previous_dim = dims[0]
        for i, dim, activation in zip(range(len(activations)),
                                      dims[1:], activations):
            with tf.variable_scope('layer' + str(i)):
                weights = weight_variable((previous_dim, dim),
                                          stddev / np.sqrt(previous_dim))
                if i < len(activations) - 1:
                    biases = bias_variable((dim,), value=bias_value)
                else:
                    biases = bias_variable((dim,), value=0.0)

            self.layers.append((weights, biases, activation))
            previous_dim = dim

    def __call__(self, x, add_bias=True, return_activations=False):
        h = x
        hidden = []
        for weights, biases, activation in self.layers:
            h = tf.matmul(h, weights)
            if add_bias:
                h += biases
            if activation:
                h = activation(h)
            hidden.append(h)
        self.hidden = hidden
        if return_activations:
            return hidden
        else:
            return h

    def get_forward_derivative(self, x, fprimes):
        h = x
        for layer, fprime in zip(self.layers, fprimes):
            weights, biases, activation = layer
            h = tf.matmul(h, weights)
            h *= fprime
        return h



class MLPBlock(object):
    """Applies a separate MLP to each dimension of the input.

    The output dimensionality is assumed to be identical to the input.
    """

    def __init__(self, input_dim, hidden_dim, n_layers=1,
            stddev=1., bias_value=0.0):
        # bias value will only be applied to the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        with tf.variable_scope('block_mlp'):
            self.w_in_var = weight_variable((input_dim, input_dim * hidden_dim),
                                   stddev / np.sqrt(hidden_dim),
                                   name='w_in')
            self.w_out_var = weight_variable((input_dim * hidden_dim, input_dim),
                                    stddev / np.sqrt(hidden_dim),
                                    name='w_out')
            mask = np.zeros((input_dim, input_dim * hidden_dim),
                            dtype='float32')
            hid_to_hid_mask = np.zeros((input_dim * hidden_dim,
                                        input_dim * hidden_dim),
                                       dtype='float32')
            self.bias_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value,
                                          name='bias_first_hid')
            self.bias_out = bias_variable((input_dim,),
                                          name='bias_out')
            for i, row in enumerate(mask):
                row[i * hidden_dim:(i + 1) * hidden_dim] = 1.0

            for i in range(0, input_dim * hidden_dim, hidden_dim):
                hid_to_hid_mask[i:i + hidden_dim, i:i + hidden_dim] = 1.0

            self.hid_to_hid_mask = tf.convert_to_tensor(hid_to_hid_mask)
            self.in_out_mask = tf.convert_to_tensor(mask)
            self.w_in = self.w_in_var * self.in_out_mask # element by element
            self.w_out = self.w_out_var * tf.transpose(self.in_out_mask) # element by element
            for i in range(n_layers - 1):
                with tf.variable_scope('layer_' + str(i)):
                    w_hid = weight_variable((input_dim * hidden_dim,
                                             input_dim * hidden_dim),
                                             stddev / np.sqrt(hidden_dim))
                    b_hid = bias_variable((hidden_dim * input_dim,),
                                          value=bias_value)
                    self.hidden_layers.append((w_hid * self.hid_to_hid_mask,
                                               b_hid))

    def __call__(self, y, **kwargs):
        return self.forward(y, **kwargs)

    def forward(self, y, activation=None):
        h = tf.matmul(y, self.w_in) + self.bias_hid
        if activation is not None:
            h = activation(h)
        for w_hid, b_hid in self.hidden_layers:
            h = tf.matmul(h, w_hid) + b_hid
            if activation is not None:
                h = activation(h)
        x = tf.matmul(h, self.w_out) + self.bias_out
        return x




    
class MLPPairs_0(object):
    """Applies a separate MLP to each pair of the input.
    The output dimensionality is assumed to be identical to the input.
    """

    def __init__(self, input_dim, hidden_dim, n_layers=1,
            stddev=1., bias_value=0.0):
        # bias value will only be applied to the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []

        num_pairs = int((input_dim * (input_dim - 1)) / 2)
        pairs = [None] * num_pairs
        ctr = 0
        for u in range(input_dim):
            for v in range(u+1, input_dim):
                pairs[ctr] = np.array((u, v))
                ctr +=1


        with tf.variable_scope('pairs_mlp'):
            self.w_in_var = weight_variable((input_dim, num_pairs * hidden_dim),
                                   stddev / np.sqrt(hidden_dim),
                                   name='w_in')
            self.w_out_var = weight_variable((num_pairs * hidden_dim, input_dim),
                                    stddev / np.sqrt(hidden_dim),
                                    name='w_out')

            
            mask = np.zeros((input_dim, num_pairs * hidden_dim), dtype='float32')
            hid_to_hid_mask = np.zeros((num_pairs * hidden_dim, num_pairs * hidden_dim), dtype='float32')

            
            self.bias_hid = bias_variable((hidden_dim * num_pairs,), value=bias_value, name='bias_first_hid')
            self.bias_out = bias_variable((input_dim,), name='bias_out')


            for i in range(0, num_pairs * hidden_dim, hidden_dim):
                hid_to_hid_mask[i:i + hidden_dim, i:i + hidden_dim] = 1.0

            for j in range(num_pairs):
                    u = pairs[j][0]
                    v = pairs[j][1]
                    mask[u, (j*hidden_dim):((j+1)*hidden_dim)] = 1.0
                    mask[v, (j*hidden_dim):((j+1)*hidden_dim)] = 1.0


            self.hid_to_hid_mask = tf.convert_to_tensor(hid_to_hid_mask)
            self.in_out_mask = tf.convert_to_tensor(mask)
            self.w_in = self.w_in_var * self.in_out_mask # element by element
            self.w_out = self.w_out_var * tf.transpose(self.in_out_mask) # element by element
            
            for i in range(n_layers - 1):
                with tf.variable_scope('layer_' + str(i)):
                    w_hid = weight_variable((num_pairs * hidden_dim,
                                             num_pairs * hidden_dim),
                                             stddev / np.sqrt(hidden_dim))
                    b_hid = bias_variable((hidden_dim * num_pairs,),
                                          value=bias_value)
                    self.hidden_layers.append((w_hid * self.hid_to_hid_mask,
                                               b_hid))

    def __call__(self, y, **kwargs):
        return self.forward(y, **kwargs)

    def forward(self, y, activation=None):
        h = tf.matmul(y, self.w_in) + self.bias_hid
        if activation is not None:
            h = activation(h)
        for w_hid, b_hid in self.hidden_layers:
            h = tf.matmul(h, w_hid) + b_hid
            if activation is not None:
                h = activation(h)
        x = tf.matmul(h, self.w_out) + self.bias_out
        return x







    
class MLPPairs_1(object):
    """Applies a separate MLP to each pair of the input.
    The output dimensionality is assumed to be identical to the input.
    """

    def __init__(self, input_dim, hidden_dim, n_layers=1,
            stddev=1., bias_value=0.0):
        # bias value will only be applied to the hidden layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []

        num_pairs = int((input_dim * (input_dim - 1)) / 2)
        pairs = [None] * num_pairs
        ctr = 0
        for u in range(input_dim):
            for v in range(u+1, input_dim):
                pairs[ctr] = np.array((u, v))
                ctr +=1


        with tf.variable_scope('pairs_mlp'):
            self.w_in_var = weight_variable((input_dim, num_pairs * hidden_dim),
                                   stddev / np.sqrt(hidden_dim),
                                   name='w_in')
            self.w_out_var = weight_variable((num_pairs * hidden_dim, num_pairs),
                                    stddev / np.sqrt(hidden_dim),
                                    name='w_out')

            
            mask_in = np.zeros((input_dim, num_pairs * hidden_dim), dtype='float32')
            mask_out = np.zeros((num_pairs, num_pairs * hidden_dim), dtype='float32')
            hid_to_hid_mask = np.zeros((num_pairs * hidden_dim, num_pairs * hidden_dim), dtype='float32')

            
            self.bias_hid = bias_variable((hidden_dim * num_pairs,), value=bias_value, name='bias_first_hid')
            self.bias_out = bias_variable((num_pairs,), name='bias_out')


            for i in range(0, num_pairs * hidden_dim, hidden_dim):
                hid_to_hid_mask[i:i + hidden_dim, i:i + hidden_dim] = 1.0

            for j in range(num_pairs):
                    u = pairs[j][0]
                    v = pairs[j][1]
                    mask_in[u, (j*hidden_dim):((j+1)*hidden_dim)] = 1.0
                    mask_in[v, (j*hidden_dim):((j+1)*hidden_dim)] = 1.0
                    mask_out[(j*hidden_dim):((j+1)*hidden_dim), (j*hidden_dim):((j+1)*hidden_dim)] = 1.0
                    mask_out[(j*hidden_dim):((j+1)*hidden_dim), (j*hidden_dim):((j+1)*hidden_dim)] = 1.0


            self.hid_to_hid_mask = tf.convert_to_tensor(hid_to_hid_mask)
            self.in_mask = tf.convert_to_tensor(mask_in)
            self.out_mask = tf.convert_to_tensor(mask_out)
            self.w_in = self.w_in_var * self.in_mask
            self.w_out = self.w_out_var * tf.transpose(self.out_mask) # CHECK
            
            for i in range(n_layers - 1):
                with tf.variable_scope('layer_' + str(i)):
                    w_hid = weight_variable((num_pairs * hidden_dim,
                                             num_pairs * hidden_dim),
                                             stddev / np.sqrt(hidden_dim))
                    b_hid = bias_variable((hidden_dim * num_pairs,),
                                          value=bias_value)
                    self.hidden_layers.append((w_hid * self.hid_to_hid_mask,
                                               b_hid))

    def __call__(self, y, **kwargs):
        return self.forward(y, **kwargs)

    def forward(self, y, activation=None):
        h = tf.matmul(y, self.w_in) + self.bias_hid
        if activation is not None:
            h = activation(h)
        for w_hid, b_hid in self.hidden_layers:
            h = tf.matmul(h, w_hid) + b_hid
            if activation is not None:
                h = activation(h)
        x = tf.matmul(h, self.w_out) + self.bias_out
        return x
