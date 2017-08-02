import tensorflow as tf
import numpy as np
from model import PSPNetModel
from preprocess import inputs

IS_TRAINING = True
NUM_CLASSES = 19
CROP_SIZE = 713

if __name__ == '__main__':
	coord = tf.train.Coordinator()

	img_batch, anno_batch = inputs(IS_TRAINING)
	
	net = PSPNetModel({'data': img_batch}, is_training=IS_TRAINING, num_classes=NUM_CLASSES+1)
	raw_output = net.get_output()
	resized_output = tf.image.resize_images(raw_output, [CROP_SIZE, CROP_SIZE])

	anno_batch = tf.squeeze(anno_batch, squeeze_dims=[3])
	anno_batch = tf.cast(anno_batch, tf.uint8)
	
	anno_batch_onehot = tf.one_hot(anno_batch, depth=256)
	labels = anno_batch_onehot[:, :, :, 0:19]
	void_label = anno_batch_onehot[:, :, :, 255:]
	gtFine = tf.concat([labels, void_label], axis=3)

	flat_gt = tf.reshape(gtFine, [-1, NUM_CLASSES+1])
	flat_prediction = tf.reshape(resized_output, [-1, NUM_CLASSES+1])

	loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_prediction, labels=flat_gt)

	reduced_loss = tf.reduce_mean(loss)
	train_op = tf.train.AdamOptimizer(1e-4).minimize(reduced_loss)
		

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	
	sess.run(init)
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	
	for i in range(2):
		print(sess.run(resized_output))
		"""losss, _ = sess.run([reduced_loss, train_op])
		print(reduced_loss)"""
			
	coord.request_stop()
	coord.join(threads)
		