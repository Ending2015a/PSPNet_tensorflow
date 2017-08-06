import numpy as np
import tensorflow as tf
import LogWriter as log
slim = tf.contrib.slim

DEFAULT_PADDING = 'VALID'

LOG_TO_FILE = None

def log_to_file(file='pyLog.log'):
    global LOG_TO_FILE
    LOG_TO_FILE = log.LogWriter(file)
    LOG_TO_FILE.open()


def log_info(msg):
    if LOG_TO_FILE:
        LOG_TO_FILE.Write(msg)
    else:
        print(msg)


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True, is_training=False, num_classes=20):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
                                                       
        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable, initializer=initializer)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        if isinstance(padding, str):
            assert padding in ('SAME', 'VALID')
        else:
            assert isinstance(padding, int)

    @layer
    def conv(self,
             input,
             k_h,  # kernel height
             k_w,  # kernel width
             c_o,  # channel output
             s_h,  # stride
             s_w,  # stride
             name,
             relu=False,
             pad=DEFAULT_PADDING,
             padding_mode='CONSTANT',
             group=1,
             biased=False): 
        log_info(name)
        # Verify that the padding is acceptable
        self.validate_padding(pad)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # get input size
        i_h = input.get_shape()[1]
        i_w = input.get_shape()[2]
        # Convolution for a given input and kernel
        padding = pad
        if isinstance(pad, int):
            pad_mat = np.array([[0,0], [pad, pad], [pad, pad], [0, 0]])
            input = tf.pad(input, paddings=pad_mat, mode=padding_mode)
            padding = 'VALID'

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], initializer=tf.zeros_initializer())
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)

            #get output size
            o_h = output.get_shape()[1]
            o_w = output.get_shape()[2]

            logs  = '    Conv: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
                .format(name, i_w, i_h, o_w, o_h)
            logs += '        size = {0} * {1}, kernels = {2}, stride = {3} * {4}, pad = {5}\n'\
                    .format(k_w, k_h, c_o, s_w, s_h, pad)
            logs += '        bias = {0}, relu = {1}\n'.format(biased, relu)
            log_info(logs)

            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=False,
                    pad=DEFAULT_PADDING,
                    padding_mode='CONSTANT',
                    group=1,
                    biased=False):
        log_info(name)
        # Verify that the padding is acceptable
        self.validate_padding(pad)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # get input size
        i_h = input.get_shape()[1]
        i_w = input.get_shape()[2]
        # Convolution for a given input and kernel
        padding = pad
        if isinstance(pad, int):
            pad_mat = np.array([[0,0], [pad, pad], [pad, pad], [0, 0]])
            input = tf.pad(input, paddings=pad_mat, mode=padding_mode)
            padding = 'VALID'
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], initializer = tf.zeros_initializer())
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)

            #get output size
            o_h = output.get_shape()[1]
            o_w = output.get_shape()[2]

            logs  = '    Atros: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
                .format(name, i_w, i_h, o_w, o_h)
            logs += '        size = {0} * {1}, kernels = {2}, pad = {3}\n'\
                    .format(k_w, k_h, c_o, pad)
            logs += '        bias = {0}, relu = {1}\n'.format(biased, relu)
            log_info(logs)

            return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, pad=DEFAULT_PADDING, padding_mode='CONSTANT'):
        log_info(name)
        self.validate_padding(pad)

        i_w = input.get_shape()[2]
        i_h = input.get_shape()[1]

        padding = pad
        if isinstance(pad, int):
            pad_mat = np.array([[0,0], [pad, pad], [pad, pad], [0, 0]])
            input = tf.pad(input, paddings=pad_mat, mode=padding_mode)
            padding = 'VALID'

        output = tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

        o_w = output.get_shape()[2]
        o_h = output.get_shape()[1]

        logs  = '    MaxP: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
                .format(name, i_w, i_h, o_w, o_h)
        logs += '        size = {0} * {1}, stride = {2} * {3}, pad = {4}\n'\
                .format(k_w, k_h, s_w, s_h, pad)
        log_info(logs)

        return output

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, pad=DEFAULT_PADDING, padding_mode='CONSTANT'):
        log_info(name)
        self.validate_padding(pad)

        i_w = input.get_shape()[2]
        i_h = input.get_shape()[1]

        padding = pad
        if isinstance(pad, int):
            pad_mat = np.array([[0,0], [pad, pad], [pad, pad], [0, 0]])
            input = tf.pad(input, paddings=pad_mat, mode=padding_mode)
            padding = 'VALID'

        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

        o_w = output.get_shape()[2]
        o_h = output.get_shape()[1]

        logs  = '    AvgP: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
                .format(name, i_w, i_h, o_w, o_h)
        logs += '        size = {0} * {1}, stride = {2} * {3}, pad = {4}\n'\
                .format(k_w, k_h, s_w, s_h, pad)
        log_info(logs)

        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    @layer
    def batch_normalization(self, input, momentum, name, is_training, epsilon=1e-5, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:

            i_w = input.get_shape()[2]
            i_h = input.get_shape()[1]

            output = tf.layers.batch_normalization(
                input,
                momentum = momentum,
                epsilon = epsilon,
                training=is_training,
                name = scope.name
                )
            if not activation_fn == None:
                output = activation_fn(output, name=scope.name+'/relu')

            o_w = output.get_shape()[2]
            o_h = output.get_shape()[1]

            logs  = '    BN: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
                .format(name, i_w, i_h, o_w, o_h)
            logs += '        momentum = {0}, epsilon = {1}\n'\
                    .format(momentum, epsilon)

            if not activation_fn == None:
                logs += '        {0}: True\n'.format(activation_fn.__name__)
            log_info(logs)

            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def concat_with_interp(self, inputs, nheight, nwidth, axis, name):
        log_info(name)
    	for i in range(len(inputs)):
    		inputs[i] = tf.image.resize_images(inputs[i], [nheight, nwidth])

        output = tf.concat(axis=axis, values=inputs, name=name)

        o_h = output.get_shape()[1]
        o_w = output.get_shape()[2]

        logs =  '    Concat & Interp: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
            .format(name, '_', '_', o_w, o_h)

        logs += '        axis = {0}\n'.format(axis)

        log_info(logs)

    	return output

    @layer
    def interp(self, input, zoom_factor, name):
        log_info(name)
        shape = input.get_shape().as_list()
        i_h = shape[1]
        i_w = shape[2]

        nh = i_h + (i_h-1) * (zoom_factor-1)
        nw = i_w + (i_w-1) * (zoom_factor-1)
        output = tf.image.resize_images(input, [nh, nw])

        logs =  '    Interp: name = {0}, input = {1} * {2}, output = {3} * {4}\n'\
            .format(name, i_w, i_h, nw, nh)
        logs += '        zoom_factor = {0}\n'.format(zoom_factor)

        log_info(logs)

        return output
