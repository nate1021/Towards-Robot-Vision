import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import uuid

import common

INPUT_DEPTH = 4
STDDEV = 0.01


def inference(x_input, is_training, blank=''):
    name_counter = 0

    def next_name():
        nonlocal name_counter
        name_counter += 1
        return blank + ('latent%d' % name_counter)

    depth_in = INPUT_DEPTH

    low_level_masks = []
    var_list = []

    dropout_rate = 0.2
    non_dropout_scaledown = 1.0 - dropout_rate

    def dropout_mask(shape):
        var = tf.get_variable(str(uuid.uuid4()), shape=shape, initializer=tf.initializers.random_uniform())
        low_level_masks.append(var)
        var_one_or_zero = tf.where(is_training,
                                   tf.where(tf.greater_equal(var, dropout_rate), tf.ones_like(var), tf.zeros_like(var)),
                                   tf.ones_like(var) * non_dropout_scaledown)
        return var_one_or_zero

    def full_layer(input, depth_out):
        nonlocal depth_in, var_list
        filter_weights = tf.get_variable(name=str(next_name()), initializer=tf.random_normal([depth_in, depth_out], stddev=STDDEV))
        var_list.append(filter_weights)
        filter_bias = tf.get_variable(name=str(next_name()), initializer=tf.zeros([depth_out]))
        var_list.append(filter_bias)
        net_convo = tf.matmul(input, filter_weights)
        net = tf.nn.bias_add(value=net_convo, bias=filter_bias)
        depth_in = depth_out
        return net

    fc1 = tf.nn.leaky_relu(full_layer(x_input, 50))
    fc1 = tf.multiply(dropout_mask([depth_in]), fc1)

    fc2 = tf.nn.leaky_relu(full_layer(fc1, 50))

    fc3 = tf.nn.leaky_relu(full_layer(fc2, 50))
    fc4 = tf.nn.leaky_relu(full_layer(fc3, 50))
    fc5 = tf.nn.leaky_relu(full_layer(fc4, 50))

    fc6 = tf.nn.leaky_relu(full_layer(fc5, 50))
    fc7 = tf.nn.leaky_relu(full_layer(fc6, 50))

    fc_final = full_layer(fc7, common.LATENT_COUNT)

    return fc_final, low_level_masks, var_list


# region Saver setup
SAVE_PATH_TEMPLATE = '../../saved_models/0/lat_vec_save_%d/'


def get_saver(position, blank=''):
    lst = [n for n in tf.global_variables() if n.name.startswith(blank + 'latent') and '/' not in n.name]
    saver = tf.train.Saver(var_list=lst)
    path_model = SAVE_PATH_TEMPLATE % position
    common.mkdir(path_model)

    def save(sess):
        saver.save(sess=sess, save_path=path_model)

    def restore(sess):
        saver.restore(sess=sess, save_path=path_model)
    return save, restore

# endregion
