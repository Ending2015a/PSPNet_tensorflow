import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

data_root = '/home/joehsiao/Dataset/cityscapes'
train_list = '/home/joehsiao/Dataset/cityscapes/train.txt'

BATCH_SIZE = 1
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
CROP_SIZE = 713
IS_TRAINING = True
NUM_CHANNELS = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def read_labeled_image_list(is_training, filelist):
	image_list = []
	anno_list = []

	if is_training:
		f = open(filelist, 'r')

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

def _generate_image_and_label_batch(image, anno, min_queue_examples, batch_size, shuffle):
	num_preprocess_threads = 16
	if shuffle:
		image_batch, anno_batch = tf.train.shuffle_batch(
				[image, anno],
				batch_size=batch_size,
				enqueue_many=True,
				capacity=1000,
				min_after_dequeue=min_queue_examples)
	else:
		image_batch, anno_batch = tf.train.batch(
				[image, anno],
				batch_size=batch_size,
				enqueue_many=True,
				capacity=min_queue_examples + 3 * batch_size)

	tf.summary.image('images', image)
	tf.summary.image('anno', anno)

	return image_batch, anno_batch

def pre_processing(image, anno):
	stride_rate = 2. / 3.
	stride = math.ceil(CROP_SIZE * stride_rate)
	
	h_grid = math.ceil(float(IMAGE_HEIGHT - CROP_SIZE) / stride) + 1
	w_grid = math.ceil(float(IMAGE_WIDTH - CROP_SIZE) / stride) + 1
	
	print("-----Information------")
	print("image size: ({h}, {w})".format(h=IMAGE_HEIGHT, w=IMAGE_WIDTH))
	print("crop size: ({c_sz}, {c_sz})".format(c_sz=CROP_SIZE))
	print("crop whole image into ({h}, {w}) grids ".format(h=h_grid, w=w_grid))
	print("-----Information------")
	
	sub_imgs = []
	sub_annos = []
	for idx_h in range(int(h_grid)):
		for idx_w in range(int(w_grid)):
			s_w = int(idx_w * stride)
			s_h = int(idx_h * stride)
			e_w = min(s_w + CROP_SIZE - 1, IMAGE_WIDTH - 1)
			e_h = min(s_h + CROP_SIZE - 1, IMAGE_HEIGHT - 1)
			s_w = e_w - CROP_SIZE + 1 
			s_h = e_h - CROP_SIZE + 1
			
			sub_imgs.append(tf.image.crop_to_bounding_box(image, s_h, s_w, CROP_SIZE, CROP_SIZE))
			sub_annos.append(tf.image.crop_to_bounding_box(anno, s_h, s_w, CROP_SIZE, CROP_SIZE))

	
	"""
	init = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		i = 1
		plt.figure()
		for img in sub_annos:
			im = img.eval()
			plt.subplot(2,4,i)
			plt.imshow(np.asarray(im, dtype=np.uint8))
			i = i + 1
			#Image.fromarray(np.asarray(im, dtype=np.uint8)).show()
		
		plt.show()

		coord.request_stop()
		coord.join(threads)
	"""	

	return sub_imgs, sub_annos

def inputs(is_training):
	image_list, anno_list = read_labeled_image_list(is_training, train_list)

	image = tf.convert_to_tensor(image_list, dtype=tf.string)
	anno = tf.convert_to_tensor(anno_list, dtype=tf.string)

	input_queue = tf.train.slice_input_producer([image, anno])
	image, anno = read_images_from_disk(input_queue)

	if is_training:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

	image = tf.cast(image, tf.float32)
	anno = tf.cast(anno, tf.float32)
	image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
	anno.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	sub_imgs, sub_annos = pre_processing(image, anno)
	
	min_queue_examples = 10 * BATCH_SIZE

	img_batch, anno_batch = _generate_image_and_label_batch(sub_imgs, sub_annos, min_queue_examples, BATCH_SIZE, shuffle=is_training)
	"""
	shape = img_batch.get_shape().as_list()
	dim = np.prod(shape[:2])
	img_batch = tf.reshape(img_batch, [dim, CROP_SIZE, CROP_SIZE, NUM_CHANNELS])
	anno_batch = tf.reshape(anno_batch, [dim, CROP_SIZE, CROP_SIZE, 1])
	"""
	img_batch = tf.subtract(img_batch, IMG_MEAN)

	print('reshaped img_batch size: ', img_batch.get_shape())
	print('reshaped anno_batch size: ', anno_batch.get_shape())

	return img_batch, anno_batch

if __name__ == "__main__":
	img_batch, anno_batch = inputs(IS_TRAINING)
	
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	print(sess.run(img_batch))

	coord.request_stop()
	coord.join(threads)
