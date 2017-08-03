import tensorflow as tf
import numpy as np
from model import PSPNetModel
from preprocess import inputs

IS_TRAINING = True
NUM_CLASSES = 19
CROP_SIZE = 713
OUT_SIZE = 90

if __name__ == '__main__':
	img_batch, anno_batch = inputs(IS_TRAINING)
	
	net = PSPNetModel({'data': img_batch}, is_training=IS_TRAINING, num_classes=NUM_CLASSES+1)
	raw_output = net.get_output()
	print('get raw output')
	print(raw_output.get_shape())
	resized_output = tf.image.resize_images(raw_output, [CROP_SIZE, CROP_SIZE])
	print('get output and resized')
	print(resized_output.get_shape())

	anno_batch = tf.image.resize_images(anno_batch, [OUT_SIZE, OUT_SIZE])
	anno_batch = tf.squeeze(anno_batch, squeeze_dims=[3])
	anno_batch = tf.cast(anno_batch, tf.uint8)
	
	anno_batch_onehot = tf.one_hot(anno_batch, depth=256)
	labels = anno_batch_onehot[:, :, :, 0:19]
	void_label = anno_batch_onehot[:, :, :, 255:]
	gtFine = tf.concat([labels, void_label], axis=3)
	print('get onehot label')
	print(gtFine.get_shape())

	flat_gt = tf.reshape(gtFine, [-1, NUM_CLASSES+1])
	flat_prediction = tf.reshape(raw_output, [-1, NUM_CLASSES+1])
	print(flat_gt.get_shape())
	print(flat_prediction.get_shape())

	cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_gt)
	cross_entropy_sum = tf.reduce_sum(cross_entropies)

	print(cross_entropies.get_shape())
	print(cross_entropy_sum.get_shape())

	optimizer = tf.train.GradientDescentOptimizer(0.5)
	trainnn = optimizer.minimize(cross_entropy_sum)
	train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_sum)
	
	print('set config')
		
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	for i in range(20):
		loss, __ = sess.run([cross_entropy_sum, trainnn])
		print(loss)
	
	coord.request_stop()
	coord.join(threads)
		