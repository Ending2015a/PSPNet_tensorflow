import tensorflow as tf
import numpy as np
import network
import os
from model import PSPNetModel
from preprocess import inputs

IS_TRAINING = True
NUM_CLASSES = 19
CROP_SIZE = 713
OUT_SIZE = 713

LEARNING_RATE = 0.0001
POWER = 0.9
MOMENTUM = 0.9
DECAY_RATE = 0.0001
MAXIMUM_ITER = 100000
CHECKPOINT_PATH = None
SNAPSHOT_DIR = './snapshots/'
train_with_resized = False

network.log_to_file()

def save(saver, sess, logdir, step):
	model_name = 'model.ckpt'
	checkpoint_path = os.path.join(logdir, model_name)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	saver.save(sess, checkpoint_path, global_step=step)
	print('Checkpoint has been created.')

def load(saver, sess, ckpt_path):
	saver.restore(sess, ckpt_path)
	print("Restored model from {}".format(ckpt_path))

if __name__ == '__main__':
    img_batch, anno_batch = inputs(IS_TRAINING)

    print('img_batch: {0}'.format(img_batch.get_shape()))
    print('anno_batch: {0}'.format(anno_batch.get_shape()))

    net = PSPNetModel({'data': img_batch}, is_training=IS_TRAINING, num_classes=NUM_CLASSES+1)

    if train_with_resized == True:
    	raw_output = net.get_output()
    else:	# calculate loss with 1/8 size
    	raw_output = net.layers['conv6']
    	anno_batch = tf.image.resize_images(anno_batch, [90, 90])

    anno_batch = tf.squeeze(anno_batch, squeeze_dims=[3])
    anno_batch = tf.cast(anno_batch, tf.uint8)

    anno_batch_onehot = tf.one_hot(anno_batch, depth=256)
    labels = anno_batch_onehot[:, :, :, 0:19]
    void_label = anno_batch_onehot[:, :, :, 255:]
    gtFine = tf.concat([labels, void_label], axis=3)

    print('raw_output: {0}'.format(raw_output.get_shape()))
    print('gtFine: {0}'.format(gtFine.get_shape()))

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=raw_output, labels=gtFine, name="softmax")
    cross_entropy_sum = tf.reduce_sum(cross_entropies)
  
    print('Set hyperparameter')

    base_lr = tf.constant(LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / MAXIMUM_ITER), POWER))

    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_sum)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    restore_var = [v for v in tf.global_variables()]
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    ckpt = tf.train.get_checkpoint_state(SNAPSHOT_DIR)
    load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

    if ckpt and ckpt.model_checkpoint_path:
    	loader = tf.train.Saver(var_list=restore_var)
    	load(loader, sess, ckpt.model_checkpoint_path)
    else:
    	print('No checkpoint file found.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(MAXIMUM_ITER):
        feed_dict = {step_ph: step + load_step}
        loss, __, lr = sess.run([cross_entropy_sum, train_step, learning_rate], feed_dict=feed_dict)
        
        if step % 100 == 0:
			print('iter {0}: loss={1}, lr: {2}'.format(step + load_step, loss, lr))
        if step % 1000 == 0:
        	save(saver, sess, SNAPSHOT_DIR, step)
        	print('iter {0}: save checkpoint'.format(step + load_step))

    coord.request_stop()
    coord.join(threads)
