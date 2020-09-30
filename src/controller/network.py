import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np

import common

LOCATION_N = 50  # number of random starting locations
CONTROLLER_N = 100  # controller count

FCi_DEPTH = [50, 50]  # test for good initial motor commands when changing this
OUTPUT_DEPTH = 2
INPUT_DEPTH = common.LATENT_COUNT

weight_placeholders = []
bias_placeholders = []
hidden_output_placeholders = []
hidden_internal_placeholders = []

for i in range(len(FCi_DEPTH) + 1):
    if i == 0:
        prev = INPUT_DEPTH
    else:
        prev = FCi_DEPTH[i - 1]

    if i == len(FCi_DEPTH):
        nxt = OUTPUT_DEPTH
    else:
        nxt = FCi_DEPTH[i]

    if i < len(FCi_DEPTH):
        for gate_i in range(len(['input','output_gate','input_gate','forget_gate'])):  # names are just for readability
            weight_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, prev, nxt], dtype=tf.float32))  # from prev layer
            weight_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, nxt, nxt], dtype=tf.float32))  # from prev time
            bias_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, 1, nxt], dtype=tf.float32))

        hidden_output_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]], dtype=tf.float32))
        hidden_internal_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]], dtype=tf.float32))

        weight_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, prev, nxt], dtype=tf.float32))
        bias_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, 1, nxt], dtype=tf.float32))
    else:
        weight_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, prev, nxt], dtype=tf.float32))
        bias_placeholders.append(tf.placeholder(shape=[CONTROLLER_N, 1, nxt], dtype=tf.float32))


def inference(input_placeholder, hidden_output_placeholders, hidden_internal_placeholders):

    hidden_output = []
    hidden_internal = []
    weight_index = 0
    bias_index = 0
    for i in range(len(FCi_DEPTH) + 1):
        if i == 0:
            prev = input_placeholder
        final_layer = i == len(FCi_DEPTH)

        if not final_layer:
            gate_input = tf.nn.sigmoid(tf.matmul(prev, weight_placeholders[weight_index + 0]) + tf.matmul(hidden_output_placeholders[i], weight_placeholders[weight_index + 1]) + bias_placeholders[bias_index + 0])
            gate_output = tf.nn.sigmoid(tf.matmul(prev, weight_placeholders[weight_index + 2]) + tf.matmul(hidden_output_placeholders[i], weight_placeholders[weight_index + 3]) + bias_placeholders[bias_index + 1])
            gate_forget = tf.nn.sigmoid(tf.matmul(prev, weight_placeholders[weight_index + 4]) + tf.matmul(hidden_output_placeholders[i], weight_placeholders[weight_index + 5]) + bias_placeholders[bias_index + 2])
            input_value = tf.nn.leaky_relu(tf.matmul(prev, weight_placeholders[weight_index + 6]) + tf.matmul(hidden_output_placeholders[i], weight_placeholders[weight_index + 7]) + bias_placeholders[bias_index + 3])

            weight_index += 8
            bias_index += 4

            internal_value = tf.multiply(gate_input, input_value) + tf.multiply(gate_forget, hidden_internal_placeholders[i])
            output_value = tf.multiply(tf.nn.leaky_relu(internal_value), gate_output)

            prev = output_value + tf.nn.leaky_relu(tf.matmul(prev, weight_placeholders[weight_index]) + bias_placeholders[bias_index])
            weight_index += 1
            bias_index += 1

            hidden_output.append(prev)
            hidden_internal.append(internal_value)

        else:
            net = tf.matmul(prev, weight_placeholders[weight_index]) + bias_placeholders[bias_index]
            weight_index += 1
            bias_index += 1
            if final_layer:
                prev = (0.9 * tf.nn.tanh(net)) + 0.1 #encorage facing the forward direction
            else:
                prev = tf.nn.leaky_relu(net)

    return prev, hidden_output, hidden_internal


gene_count = 0
for w in weight_placeholders:
    gene_count += np.product(w.get_shape().as_list()[1:None])
for b in bias_placeholders:
    gene_count += np.product(b.get_shape().as_list()[1:None])


def first_hidden_state_tensor():
    hidden_output = []
    hidden_internal = []
    for i in range(len(FCi_DEPTH)):
        hidden_output.append(tf.zeros(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]]))
        hidden_internal.append(tf.zeros(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]]))
    return hidden_output, hidden_internal


def first_hidden_state_numpy():
    hidden_output = []
    hidden_internal = []
    for i in range(len(FCi_DEPTH)):
        hidden_output.append(np.zeros(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]]))
        hidden_internal.append(np.zeros(shape=[CONTROLLER_N, LOCATION_N, FCi_DEPTH[i]]))
    return hidden_output, hidden_internal


def gen_feed_dict(population):
    feed_dict = {}
    gene_pos = 0
    for iw in range(len(weight_placeholders)):
        w = weight_placeholders[iw]
        w_count = np.product(w.get_shape().as_list()[1:None])

        feed_dict[w] = population[:, gene_pos:gene_pos + w_count].reshape(w.get_shape().as_list())
        gene_pos += w_count

    for ib in range(len(bias_placeholders)):
        b = bias_placeholders[ib]
        b_count = np.product(b.get_shape().as_list()[1:None])

        feed_dict[b] = population[:, gene_pos:gene_pos + b_count].reshape(b.get_shape().as_list())
        gene_pos += b_count

    if gene_pos != gene_count or gene_count != population.shape[1]:
        input('Unit test fail!')

    return feed_dict
