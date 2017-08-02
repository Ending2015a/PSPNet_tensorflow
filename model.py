# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from network import Network
import tensorflow as tf

class PSPNetModel(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, name='conv1_1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv1_1_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv1_2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv1_2_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv1_3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv1_3_bn')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_proj')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_1_proj_bn'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_1_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_1_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_1_increase_bn'))

        (self.feed('conv2_1_proj_bn', 
                   'conv2_1_increase_bn')
             .add(name='conv2_1_add')
             .relu(name='conv2_1_add_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_2_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_2_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_2_increase_bn'))

        (self.feed('conv2_1_add_relu', 
                   'conv2_2_increase_bn')
             .add(name='conv2_2_add')
             .relu(name='conv2_2_add_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_3_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv2_3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_3_increase_bn'))

        (self.feed('conv2_2_add_relu', 
                   'conv2_3_increase_bn')
             .add(name='conv2_3_add')
             .relu(name='conv2_3_add_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_proj')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_1_proj_bn'))

        (self.feed('conv2_3_add_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_1_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_1_reduce_bn')
             .conv(3, 3, 128, 2, 2, biased=False, relu=False, name='conv3_1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_1_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_1_increase_bn'))

        (self.feed('conv3_1_proj_bn', 
                   'conv3_1_increase_bn')
             .add(name='conv3_1_add')
             .relu(name='conv3_1_add_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_2_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_2_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_2_increase_bn'))

        (self.feed('conv3_1_add_relu', 
                   'conv3_2_increase_bn')
             .add(name='conv3_2_add')
             .relu(name='conv3_2_add_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_3_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_3_increase_bn'))

        (self.feed('conv3_2_add_relu', 
                   'conv3_3_increase_bn')
             .add(name='conv3_3_add')
             .relu(name='conv3_3_add_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_4_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv3_4_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_4_increase_bn'))

        (self.feed('conv3_3_add_relu', 
                   'conv3_4_increase_bn')
             .add(name='conv3_4_add')
             .relu(name='conv3_4_add_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_proj')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_1_proj_bn'))

        (self.feed('conv3_4_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_1_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_1_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_1_increase_bn'))

        (self.feed('conv4_1_proj_bn', 
                   'conv4_1_increase_bn')
             .add(name='conv4_1_add')
             .relu(name='conv4_1_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_2_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_2_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_2_increase_bn'))

        (self.feed('conv4_1_add_relu', 
                   'conv4_2_increase_bn')
             .add(name='conv4_2_add')
             .relu(name='conv4_2_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_3_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_3_increase_bn'))

        (self.feed('conv4_2_add_relu', 
                   'conv4_3_increase_bn')
             .add(name='conv4_3_add')
             .relu(name='conv4_3_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_4_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_4_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_4_increase_bn'))

        (self.feed('conv4_3_add_relu', 
                   'conv4_4_increase_bn')
             .add(name='conv4_4_add')
             .relu(name='conv4_4_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_5_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_5')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_5_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_5_increase_bn'))

        (self.feed('conv4_4_add_relu', 
                   'conv4_5_increase_bn')
             .add(name='conv4_5_add')
             .relu(name='conv4_5_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_6_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_6')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_6_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_6_increase_bn'))

        (self.feed('conv4_5_add_relu', 
                   'conv4_6_increase_bn')
             .add(name='conv4_6_add')
             .relu(name='conv4_6_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_7_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_7_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_7')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_7_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_7_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_7_increase_bn'))

        (self.feed('conv4_6_add_relu', 
                   'conv4_7_increase_bn')
             .add(name='conv4_7_add')
             .relu(name='conv4_7_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_8_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_8_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_8')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_8_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_8_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_8_increase_bn'))

        (self.feed('conv4_7_add_relu', 
                   'conv4_8_increase_bn')
             .add(name='conv4_8_add')
             .relu(name='conv4_8_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_9_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_9_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_9')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_9_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_9_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_9_increase_bn'))

        (self.feed('conv4_8_add_relu', 
                   'conv4_9_increase_bn')
             .add(name='conv4_9_add')
             .relu(name='conv4_9_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_10_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_10_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_10')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_10_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_10_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_10_increase_bn'))

        (self.feed('conv4_9_add_relu', 
                   'conv4_10_increase_bn')
             .add(name='conv4_10_add')
             .relu(name='conv4_10_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_11_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_11_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_11')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_11_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_11_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_11_increase_bn'))

        (self.feed('conv4_10_add_relu', 
                   'conv4_11_increase_bn')
             .add(name='conv4_11_add')
             .relu(name='conv4_11_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_12_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_12_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_12')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_12_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_12_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_12_increase_bn'))

        (self.feed('conv4_11_add_relu', 
                   'conv4_12_increase_bn')
             .add(name='conv4_12_add')
             .relu(name='conv4_12_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_13_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_13_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_13')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_13_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_13_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_13_increase_bn'))

        (self.feed('conv4_12_add_relu', 
                   'conv4_13_increase_bn')
             .add(name='conv4_13_add')
             .relu(name='conv4_13_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_14_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_14_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_14')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_14_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_14_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_14_increase_bn'))

        (self.feed('conv4_13_add_relu', 
                   'conv4_14_increase_bn')
             .add(name='conv4_14_add')
             .relu(name='conv4_14_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_15_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_15_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_15')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_15_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_15_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_15_increase_bns'))

        (self.feed('conv4_14_add_relu', 
                   'conv4_15_increase_bns')
             .add(name='conv4_15_add')
             .relu(name='conv4_15_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_16_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_16_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_16')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_16_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_16_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_16_increase_bn'))

        (self.feed('conv4_15_add_relu', 
                   'conv4_16_increase_bn')
             .add(name='conv4_16_add')
             .relu(name='conv4_16_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_17_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_17_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_17')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_17_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_17_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_17_increase_bn'))

        (self.feed('conv4_16_add_relu', 
                   'conv4_17_increase_bn')
             .add(name='conv4_17_add')
             .relu(name='conv4_17_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_18_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_18_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_18')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_18_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_18_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_18_increase_bn'))

        (self.feed('conv4_17_add_relu', 
                   'conv4_18_increase_bn')
             .add(name='conv4_18_add')
             .relu(name='conv4_18_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_19_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_19_reduce_bns')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_19')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_19_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_19_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_19_increase_bn'))

        (self.feed('conv4_18_add_relu', 
                   'conv4_19_increase_bn')
             .add(name='conv4_19_add')
             .relu(name='conv4_19_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_20_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_20_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_20')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_20_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_20_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_20_increase_bn'))

        (self.feed('conv4_19_add_relu', 
                   'conv4_20_increase_bn')
             .add(name='conv4_20_add')
             .relu(name='conv4_20_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_21_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_21_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_21')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_21_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_21_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_21_increase_bn'))

        (self.feed('conv4_20_add_relu', 
                   'conv4_21_increase_bn')
             .add(name='conv4_21_add')
             .relu(name='conv4_21_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_22_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_22_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_22')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_22_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_22_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_22_increase_bn'))

        (self.feed('conv4_21_add_relu', 
                   'conv4_22_increase_bn')
             .add(name='conv4_22_add')
             .relu(name='conv4_22_add_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_23_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv4_23_reduce_bn')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='atrous4_23')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous4_23_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_23_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_23_increase_bn'))

        (self.feed('conv4_22_add_relu', 
                   'conv4_23_increase_bn')
             .add(name='conv4_23_add')
             .relu(name='conv4_23_add_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_proj')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_1_proj_bn'))

        (self.feed('conv4_23_add_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_1_reduce_bn')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='atrous5_1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous5_1_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_1_increase_bn'))

        (self.feed('conv5_1_proj_bn', 
                   'conv5_1_increase_bn')
             .add(name='conv5_1_add')
             .relu(name='conv5_1_add_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_2_reduce_bn')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='atrous5_2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='atrous5_2_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_2_increase_bn'))

        (self.feed('conv5_1_add_relu', 
                   'conv5_2_increase_bn')
             .add(name='conv5_2_add')
             .relu(name='conv5_2_add_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_reduce')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_3_reduce_bn')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='atrous5_3')
             .batch_normalization(activation_fn=tf.nn.relu, name='atrous5_3_bn', is_training=is_training)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_increase')
             .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_3_increase_bn'))

        (self.feed('conv5_2_add_relu', 
                   'conv5_3_increase_bn')
             .add(name='conv5_3_add')
             .relu(name='conv5_3_add_relu')
             .avg_pool(90, 90, 90, 90, name='conv5_3_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool1_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_3_pool1_conv_bn'))

        (self.feed('conv5_3_add_relu')
             .avg_pool(45, 45, 45, 45, name='conv5_3_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool2_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_3_pool2_conv_bn'))

        (self.feed('conv5_3_add_relu')
             .avg_pool(30, 30, 30, 30, name='conv5_3_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool3_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_3_pool3_conv_bn'))

        (self.feed('conv5_3_add_relu')
             .avg_pool(15, 15, 15, 15, name='conv5_3_pool4')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_pool4_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_3_pool4_conv_bn'))

        (self.feed('conv5_3_add_relu', 
                   'conv5_3_pool1_conv_bn', 
                   'conv5_3_pool2_conv_bn', 
                   'conv5_3_pool3_conv_bn',
                   'conv5_3_pool4_conv_bn')
             .concat_with_interp(90, 90, 3, name='conv5_3_concat'))

        (self.feed('conv5_3_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv5_4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='conv5_4_bn')
             .dropout(0.1, name='conv5_4_dropout'))

        (self.feed('conv5_4_dropout')
             .conv(1, 1, num_classes, 1, 1, biased=False, relu=False, name='conv6'))