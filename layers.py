# not use

import numpy as np
import tensorflow as tf

class PSPNet:
    def __init__(self, input):
        self.networks = {}
        self.last_layer = None
        self.count = 0
        self.build_networks(input)

    def build_networks(self, input):
        #self.data = tf.placeholder(dtype=tf.float32, shape=[None, 713, 713, 3])

        # conv_1
        self.conv_block(64, 3, 2, 1, 0.95, 1e-5, name='conv1_1_3x3_s2', input=input)
        self.conv_block(64, 3, 1, 1, 0.95, 1e-5, name='conv1_2_3x3')
        self.conv_block(128, 3, 1, 1, 0.95, 1e-5, name='conv1_3_3x3')

        # conv_2
        self.pool_block(3, 2, 1, name='pool1_3x3_s2')
        self.conv_block(64, 1, 1, 0, 0.95, 1e-5, name='conv2_1_1x1_reduce')
        self.conv_block(64, 3, 1, 1, 0.95, 1e-5, name='conv2_1_3x3')
        self.conv_block(256, 1, 1, 0, 0.95, 1e-5, name='conv2_1_1x1_increase', relu=False)

        self.conv_block(256, 1, 1, 0, 0.95, 1e-5, name='conv2_1_1x1_proj', relu=False, input='pool1_3x3_s2')
        self.


    def sum_block(self, *args, relu=True):
        count = len(args)
        for i in range(count):


    def conv_block(self, kernels, size, stride, pad, 
        momentum, epsilon, name, padding_mode='CONSTANT', input=None, bn=True, relu=True):

        if input == None:
            input = self.last_layer
        else if isinstance(input, str):
            input = self.networks[input]

        # check
        assert len(input.get_shape()) == 4
        assert padding_mode in ('CONSTANT', 'REFLECT', 'SYMMETRIC')

        # get input shapes
        _, input_w, input_h, input_cn = input.get_shape()

        # create weights
        weights = tf.get_variable(name=name+'/weights', shape=[size, size, int(input_cn), kernels],
            dtype=input.dtype, initializer=tf.contrib.layers.xavier_initializer())

        # padding mat & padding
        pad_mat = np.array([[0,0], [pad, pad], [pad, pad], [0, 0]])
        input_pad = tf.pad(input, paddings=pad_mat, mode=padding_mode, name=name+'/pad')

        # create conv layer & bn & relu
        strides = [1, stride, stride, 1]
        conv = tf.nn.con2d(input_pad, weights, strides=strides, padding='VALID', name=name)
        self.networks[name] = conv
        if bn:
            conv_bn = tf.layers.batch_normalization(conv, momentum=momentum, epsilon=epsilon, name=name+'/bn')
            self.networks[name+'/bn'] = conv_bn
            log_bn = '    BN: Momentum = {0}, epsilon = {1}\n'.format(momentum, epsilon)
        else:
            conv_bn = conv
            log_bn = '    BN: None\n'

        if relu:
            conv_relu = tf.nn.relu(conv_bn, name=name+'/relu')
            self.networks[name+'/relu'] = conv_relu
            log_relu = '    ReLU: Yes\n'
        else:
            conv_relu = conv_bn
            log_relu = '    ReLU: None\n'

        # get output shape
        _, output_w, output_h, _ = conv_relu.get_shape()

        # print log
        logs  = 'Conv Block: name = {0}, Input = {1} * {2}, Output = {3} * {4}\n'\
                .format(name, input_w, input_h, output_w, output_h)
        logs += '    Conv: Size = {0} * {0}, Kernels = {1}, Stride = {2}, Pad = {3}\n'\
                .format(size, kernels, stride, pad)
        logs += log_bn
        logs += log_relu

        print(logs)

        self.last_layer = conv_relu

    def pool_block(self,size, stride, pad, name, type='max', padding_mode='CONSTANT'):

        input = self.last_layer

        # check
        assert len(input.get_shape()) == 4
        assert type in ('max', 'avg')
        assert padding_mode in ('CONSTANT', 'REFLECT', 'SYMMETRIC')

        # get input shape
        _, input_w, input_h, _ = input.get_shape()

        # padding
        pad_mat = np.array([[0,0],[pad,pad],[pad,pad],[0,0]])
        input_pad = tf.pad(input, paddings=pad_mat, mode=padding_mode, name=name+'/pad')

        # pooling
        ksize = [1, size, size, 1]
        strides = [1, stride, stride, 1]
        if type == 'max':
            pool = tf.nn.max_pool(input_pad, ksize=ksize, strides=strides, padding='VALID', name=name)
        else if type == 'avg':
            pool = tf.nn.avg_pool(input_pad, ksize=ksize, strides=strides, padding='VALID', name=name)

        self.networks[name] = pool

        # get output shape
        _, output_w, output_h, _ = pool.get_shape()

        logs = 'Pool Block: name={0}, Input = {1} * {2}, Output = {3} * {4}\n'\
                .format(name, input_w, input_h, output_w, output_h)
        logs += '    {0}: Size = {1} * {1}, Stride = {2}, Pad = {3}\n'\
                .format(type, size, stride, pad)

        print(logs)

        self.last_layer = pool

