import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf

import common

INPUT_DEPTH = 2
STDDEV = 0.01


def inference(x_input):
    name_counter = 0

    def next_name():
        nonlocal name_counter
        name_counter += 1
        return 'movement_%d' % name_counter

    # batch_size, image_width, image_height, image_depth = tf.shape(images)
    depth_in = INPUT_DEPTH

    def full_layer(input_t, depth_out):
        nonlocal depth_in
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([depth_in, depth_out], stddev=STDDEV))
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depth_out]))
        net_convo = tf.matmul(input_t, filter_weights)
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        depth_in = depth_out
        return net

    fc1 = tf.nn.leaky_relu(full_layer(x_input, 50))
    fc2 = tf.nn.leaky_relu(full_layer(fc1, 50))
    fc3 = tf.nn.leaky_relu(full_layer(fc2, 50))
    fc4 = tf.nn.tanh(full_layer(fc3, 3))

    return fc4


def get_saver():
    lst = [n for n in tf.global_variables() if n.name.startswith('movement_') and '/' not in n.name]
    saver = tf.train.Saver(var_list=lst)
    path_model = '../../saved_models/0/movement_save/'
    common.mkdir(path_model)

    def save(sess):
        saver.save(sess=sess, save_path=path_model)

    def restore(sess):
        saver.restore(sess=sess, save_path=path_model)
    return save, restore
