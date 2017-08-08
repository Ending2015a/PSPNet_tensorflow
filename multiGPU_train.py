import tensorflow as tf
import numpy as np
import os
import re

import network
from model import PSPNetModel
from preprocess import inputs, Preprocessor

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './checkpoint')
tf.app.flags.DEFINE_integer('max_iter', 100000)
tf.app.flags.DEFINE_integer('num_gpus', 1)
tf.app.flags.DEFINE_string('input_list', '/data/cityscapes_dataset/cityscape/list/train_list.txt')
tf.app.flags.DEFINE_string('root_dir', '/data/cityscapes_dataset/cityscape')
tf.app.flags.DEFINE_float('learning_rate', 1e-4)

TOWER_NAME = 'pspn_Dory_Tower'

POWER = 0.9
MOMENTUM = 0.9
DECAY_RATE = 0.0001
BATCH_SIZE = 1
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
CROP_SIZE = 713
IMAGE_NUM_CHANNELS = 3
LABEL_NUM_CHANNELS = 1
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)



def read_namelist_from_file():
    filelist = FLAGS.input_list
    data_root = FLAGS.root_dir

    image_list = []
    anno_list = []

    f = open(filelist, 'r')
    print('open filelist from {0}'.format(filelist))
    
    for line in f:
        img_name, anno_name = line[:-1].split(' ')
        img_name = os.path.join(data_root, img_name)
        anno_name = os.path.join(data_root, anno_name)
        
        if not tf.gfile.Exists(img_name):
            raise ValueError('Failed to find file: ' + img_name)

        if not tf.gfile.Exists(anno_name):
            raise ValueError('Failed to find file: ' + anno_name)

        image_list.append(img_name)
        anno_list.append(anno_name)
    
    f.close()

    return image_list, anno_list

def read_images_from_disk(input_queue):
    image_file = tf.read_file(input_queue[0])
    anno_file = tf.read_file(input_queue[1])
    image = tf.image.decode_image(image_file)
    anno = tf.image.decode_image(anno_file)

    return image, anno

def get_input_queue():
    image_list, anno_list = read_namelist_from_file()

    image = tf.convert_to_tensor(image_list, dtype=tf.string)
    anno = tf.convert_to_tensor(anno_list, dtype=ty.string)

    input_queue = tf.train.slice_input_producer([image, anno])
    image, anno = read_images_from_disk(input_queue)

    image = tf.cast(image, tf.float32)
    anno = tf.cast(image, tf.float32)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NUM_CHANNELS])
    anno.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_NUM_CHANNELS])

    min_queue_examples = 10 * BATCH_SIZE

    img_batch, anno_batch = tf.train.shuffle_batch(
        [image, anno],
        batch_size = BATCH_SIZE,
        enqueue_many=True,
        capacity=1000,
        min_after_dequeue=min_queue_examples)

    tf.summary.image('image', image)
    tf.summary.image('anno', image)

    return image_batch, anno_batch




def poly_lr_policy(step):

    base_lr = tf.constant(FLAGS.learning_rate)

    # poly
    lr = tf.scalar_mul(base_lr, tf.pow((1 - step / FLAGS.max_iter), POWER))
    return lr

def onehot_encoder(label):
    # 0-18 -> classes, 255 -> void label
    label = tf.squeeze(label, squeeze_dims=[3])
    label = tf.cast(label, tf.uint8)

    onehot_label = tf.one_hot(label, depth=256)
    labels = onehot_label[:, :, :, 0:19]
    void_label = onehot_label[:, :, :, 255:]
    gtFine = tf.concat([labels, void_label], axis=3)

    return gtFine


def tower_loss(scope, image, anno):

    # preprocess image
    pre = Preprocessor(crop_size=CROP_SIZE, random_scale=[0.5, 2.0])
    # set image and anno image (both are 4D tensor)
    pre.set_input(image, anno)

    with tf.variable_scope('pspn_piece') as net_scope:

        for i in range(pre.preprocess()):
            crop_image, crop_anno = pre.get_product()
            # create network
            net = PSPNetModel({'data': crop_image}, is_training=True, num_classes=NUM_CLASSES+1)

            # calculate loss
            raw_output = net.get_output()
            onehot_labels = onehot_encoder(anno)

            # check
            print('raw_output shape: {0}'.format(raw_output.get_shape()))
            print('onehot_labels shape: {0}'.format(onehot_labels.get_shape()))

            cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=raw_output, labels=onehot_labels, name='softmax')
            cross_entropy_sum - tf.reduce_sum(cross_entropies)

            # add to collection
            tf.add_to_collection('losses', cross_entropy_sum)

            # reuse variable
            tf.get_variable_scope().reuse_variables()


    # get all losses from collection
    losses = tf.get_collection('losses', scope)

    # calculate summation
    total_loss = tf.add_n(losses, name='total_loss')

    # add to summary
    for l in losses + [total_loss]:
        loss_name = re.sub('{0}_[0-9]*/'.format(TOWER_NAME), '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss



def train():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # step.1: create global step
        global_step = tf.get_variable('global_step', [1], initializer=tf.constant_initializer(0), trainable=False)

        # step.2: create learning rate policy
        lr = poly_lr_policy(global_step)

        # step.3: create optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # step.4: create inputs tensor and prefetch_queue
        image_batch, anno_batch = get_input_queue()
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [image_batch, anno_batch], capacity=2*FLAGS.num_gpus)

        # save gradients for every tower
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:{0}'.format(i)):
                    with tf.name_scope('{0}_{1}'.format(TOWER_NAME, i)) as scope:
                        # step.1: dequeue one batch for this tower
                        #    this batch size is related to the original shuffle_batch
                        image_batch, anno_batch = batch_queue.dequeue()

                        # step.2: calcuate the loss of this tower
                        #    also create network in this function
                        loss = tower_loss(scope, image_batch, anno_batch)

                        tf.get_variable_scope().reuse_variables()

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)




def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()