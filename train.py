import tensorflow as tf
import numpy as np
from model import PSPNetModel
from preprocess import inputs

IS_TRAINING = True
NUM_CLASSES = 19
CROP_SIZE = 713
OUT_SIZE = 90
LEARNING_RATE = 0.0001
POWER = 0.9
MOMENTUM = 0.9
DECAY_RATE = 0.0001
MAXIMUM_ITER = 100000


if __name__ == '__main__':
	img_batch, anno_batch = inputs(IS_TRAINING)
	
	net = PSPNetModel({'data': img_batch}, is_training=IS_TRAINING, num_classes=NUM_CLASSES+1)
	raw_output = net.get_output()
	resized_output = tf.image.resize_images(raw_output, [CROP_SIZE, CROP_SIZE])

	anno_batch = tf.image.resize_images(anno_batch, [OUT_SIZE, OUT_SIZE])
	anno_batch = tf.squeeze(anno_batch, squeeze_dims=[3])
	anno_batch = tf.cast(anno_batch, tf.uint8)
	
	anno_batch_onehot = tf.one_hot(anno_batch, depth=256)
	labels = anno_batch_onehot[:, :, :, 0:19]
	void_label = anno_batch_onehot[:, :, :, 255:]
	gtFine = tf.concat([labels, void_label], axis=3)

	cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=raw_output, labels=gtFine, name="softmax")
	cross_entropy_sum = tf.reduce_sum(cross_entropies)

	base_lr = tf.constant(LEARNING_RATE)
	step_ph = tf.placeholder(dtype=tf.float32, shape=())
	learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / MAXIMUM_ITER), POWER))

	train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_sum)
	
	all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]

	
	"""
	train_op = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(cross_entropy_sum)
	"""
	print('set config')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	for i in range(MAXIMUM_ITER):
		feed_dict = {step_ph: i}
		loss, __, lr = sess.run([cross_entropy_sum, train_step, learning_rate], feed_dict=feed_dict)
		
		print('iter {0}: loss={1}, lr: {2}'.format(i, loss, lr))
	
	coord.request_stop()
	coord.join(threads)
		
