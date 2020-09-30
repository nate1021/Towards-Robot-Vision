import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf

import common

STDDEV = 0.01


def encoder(images):

    name_counter = 0
    def next_name():
        nonlocal name_counter
        name_counter += 1
        return 'frame_encoder_%d' % name_counter

    # batch_size, image_width, image_height, image_depth = tf.shape(images)
    depthIn = common.IMAGE_DEPTH

    def convo_layer(input, depthOut, windowSize, windowStride):
        nonlocal depthIn
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([windowSize, windowSize, depthIn, depthOut], stddev=STDDEV))
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depthOut]))
        net_convo = tf.nn.conv2d(input=input, filter=filter_weights, strides=[1, windowStride, windowStride, 1], padding='SAME')
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        depthIn = depthOut
        return net

    def full_layer(input, depthIn, depthOut):
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([depthIn, depthOut], stddev=STDDEV))
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depthOut]))
        net_convo = tf.matmul(input, filter_weights)
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        return net

    conv1 = tf.nn.leaky_relu(convo_layer(input=images, depthOut=64, windowSize=5, windowStride=2))  # outputshape [batch, inheight/2, inheight/2, ]
    conv2 = tf.nn.leaky_relu(convo_layer(input=conv1,  depthOut=64, windowSize=3, windowStride=1))  # outputshape [batch, inheight/2, inheight/2, ]
    conv3 = tf.nn.leaky_relu(convo_layer(input=conv2, depthOut=128, windowSize=5, windowStride=2))  # outputshape [batch, inheight/4, inheight/4, ]
    conv4 = tf.nn.leaky_relu(convo_layer(input=conv3, depthOut=128, windowSize=3, windowStride=1))  # outputshape [batch, inheight/4, inheight/4, ]
    conv5 = tf.nn.leaky_relu(convo_layer(input=conv4, depthOut=256, windowSize=5, windowStride=2))  # outputshape [batch, inheight/8, inheight/8, ]
    conv6 = tf.nn.leaky_relu(convo_layer(input=conv5, depthOut=256, windowSize=3, windowStride=1))  # outputshape [batch, inheight/8, inheight/8, ]
    FINAL_IMAGE_WIDTH = int(common.IMAGE_WIDTH / 8)
    FINAL_IMAGE_HEIGHT = int(common.IMAGE_HEIGHT / 8)
    final_image_depth = depthIn
    depthIn = final_image_depth * FINAL_IMAGE_WIDTH * FINAL_IMAGE_HEIGHT
    flaty = tf.reshape(conv6, [-1, depthIn])
    fc1 = tf.nn.leaky_relu(full_layer(flaty, depthIn, depthIn))
    log_std = full_layer(fc1, depthIn, common.LATENT_COUNT)
    mean = full_layer(fc1, depthIn, common.LATENT_COUNT)
    normal_sample = tf.random_normal(mean=0.0,stddev=1.0,shape=[tf.shape(mean)[0], common.LATENT_COUNT])
    z = mean + (tf.exp(log_std) * normal_sample)

    return z, mean, log_std, (FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, final_image_depth)

def decoder(z, final_image):
    name_counter = 0
    def next_name():
        nonlocal name_counter
        name_counter += 1
        return 'frame_decoder_%d' % name_counter

    # batch_size, image_width, image_height, image_depth = tf.shape(images)
    FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, final_image_depth = final_image
    depthIn = final_image_depth

    in_width = FINAL_IMAGE_WIDTH
    in_height = FINAL_IMAGE_HEIGHT

    def convo_transpose_layer(input, depthOut, windowSize, windowStride):
        nonlocal depthIn, in_width, in_height
        in_width *= windowStride
        in_height *= windowStride
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([windowSize, windowSize, depthOut, depthIn], stddev=STDDEV))
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depthOut]))
        net_convo = tf.nn.conv2d_transpose(value=input, filter=filter_weights, strides=[1, windowStride, windowStride, 1], padding='SAME', output_shape=tf.stack([tf.shape(input)[0], in_height, in_width, depthOut]))
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        depthIn = depthOut
        return net

    def full_layer(input, depthIn, depthOut):
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([depthIn, depthOut], stddev=STDDEV))
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depthOut]))
        net_convo = tf.matmul(input, filter_weights)
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        return net

    fc = tf.nn.leaky_relu(full_layer(z, common.LATENT_COUNT, FINAL_IMAGE_HEIGHT * FINAL_IMAGE_WIDTH * final_image_depth))
    reform_fc = tf.reshape(fc, [-1, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, final_image_depth])

    conv1 = tf.nn.leaky_relu(convo_transpose_layer(input=reform_fc, depthOut=128, windowSize=5, windowStride=2))
    conv2 = tf.nn.leaky_relu(convo_transpose_layer(input=conv1,  depthOut=128, windowSize=3, windowStride=1))
    conv3 = tf.nn.leaky_relu(convo_transpose_layer(input=conv2, depthOut=64, windowSize=5, windowStride=2))
    conv4 = tf.nn.leaky_relu(convo_transpose_layer(input=conv3, depthOut=64, windowSize=3, windowStride=1))
    conv5 = tf.nn.tanh(convo_transpose_layer(input=conv4, depthOut=3, windowSize=5, windowStride=2))
    #conv6 = tf.nn.tanh(convo_transpose_layer(input=conv5, depthOut=3, windowSize=3, windowStride=1))

    return conv5


# Saver setup ABOVE the training section
def get_saver():
    lst = [n for n in tf.all_variables() if n.name.startswith('frame_') and '/' not in n.name]
    saver = tf.train.Saver(var_list=lst)
    path_model = '../../saved_models/0/vae_image_save/'
    common.mkdir(path_model)

    def save(sess):
        saver.save(sess=sess, save_path=path_model)

    def restore(sess):
        print('Restoring: %s' % path_model)
        saver.restore(sess=sess, save_path=path_model)

    return save, restore
