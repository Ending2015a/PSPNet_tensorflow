import tensorflow as tf
import numpy as np
import network
import os
from model import PSPNetModel
from preprocess import inputs
from PIL import Image
import matplotlib.pyplot as plt

IS_TRAINING = False
NUM_CLASSES = 19
CROP_SIZE = 713
OUT_SIZE = 713
BATCH_SIZE = 8

LEARNING_RATE = 0.0001
POWER = 0.9
MOMENTUM = 0.9
DECAY_RATE = 0.0001
MAXIMUM_ITER = 200000
CHECKPOINT_PATH = None
SNAPSHOT_DIR = './snapshots/'
SAVE_DIR = './output/'
train_with_resized = False

# colour map
label_colours = [(0, 0, 192)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,64,128),(255,255,255)]

network.log_to_file()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model from {}".format(ckpt_path))

def predict(img_batch):
    shape = img_batch.get_shape().as_list()

    prediction_list = []
    with tf.variable_scope('pspent') as scope:
        for i in range(shape[0]):
            sub_img = tf.expand_dims(img_batch[i], 0)
            net = PSPNetModel({'data': sub_img}, is_training=IS_TRAINING, num_classes=NUM_CLASSES+1)
            raw_output = net.get_output()
            prediction = tf.argmax(raw_output, axis=3)
            scope.reuse_variables()
            prediction_list.append(prediction)

    output = tf.concat(prediction_list, axis=0)
    return output

def decode_labels(mask, numofClasses):
    n, h, w = mask.shape

    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < n:
                    pixels[k_, j_] = label_colours[k]

        outputs[i] = np.array(img)

    return outputs

def show_image(imgs, index):
    n, h, w, c = imgs.shape

    plt.figure(index)
    for i in range(n):
        plt.subplot(2,4,i+1)

        im = imgs[i]
        if c == 1:
            im = np.squeeze(im, axis=2)
            plt.imshow(np.asarray(im, dtype=np.uint8), cmap='gray')
        else:
            plt.imshow(np.asarray(im, dtype=np.uint8))
        #Image.fromarray(np.asarray(im, dtype=np.uint8)).show()
        
    plt.show()

if __name__ == '__main__':
    img_batch, label_batch, numOfgrids = inputs(IS_TRAINING, batch_size=BATCH_SIZE)

    print('image batch shape: {0}'.format(img_batch.get_shape().as_list()))
    print('label batch shape: {0}'.format(label_batch.get_shape().as_list()))
    print('number of grids: {0}'.format(np.prod(numOfgrids)))

    prediction = predict(img_batch)
    print('output shape: {0}'.format(prediction.get_shape()))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Try to restore from checkpoint...')
    restore_var = tf.global_variables()
    ckpt = tf.train.get_checkpoint_state(SNAPSHOT_DIR)
    load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    _preds, img, label = sess.run([prediction, img_batch, label_batch])
    msk = decode_labels(_preds, NUM_CLASSES+1)

    for i in range(BATCH_SIZE):
        im = Image.fromarray(msk[i])
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        filename = 'mask' + str(i) + '.png'
        im.save(SAVE_DIR + filename)

    show_image(img, 0)
    show_image(label, 1)

    coord.request_stop()
    coord.join(threads)