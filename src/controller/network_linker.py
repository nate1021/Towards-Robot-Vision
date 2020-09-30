import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import math

import common
import controller.network
import latent_sim.network
import movement_sim.network

simulator_latent_vector_restore = []


def restore_models(trackers_arr, sess):
    global simulator_latent_vector_restore
    for i in range(len(trackers_arr)):
        _, restore = latent_sim.network.get_saver(trackers_arr[i]['position'])
        simulator_latent_vector_restore.append(restore)

    _, restore = movement_sim.network.get_saver()
    restore(sess)
    print('Simulator models restored.')


def load_latent_simulator(i, sess):
    simulator_latent_vector_restore[i](sess)


# global_position = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2])  # global x, global y
# global_orientation = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N])  # global bearing
def next_position_orientation(global_position, global_orientation, state, latent_noise):
    # RUN latent vec sim
    global_position_x, global_position_y = tf.unstack(global_position, axis=2)
    sn = tf.sin(global_orientation / 180 * math.pi)
    cs = tf.cos(global_orientation / 180 * math.pi)
    global_po_preprocessed = tf.stack([global_position_x, global_position_y, sn, cs], axis=2)
    global_po_preprocessed = tf.reshape(global_po_preprocessed, shape=[controller.network.CONTROLLER_N * controller.network.LOCATION_N, 4])

    simulator_latent_vector_logits, _, _ = latent_sim.network.inference(global_po_preprocessed, is_training=tf.constant(False))
    reshaped_zs = tf.reshape(simulator_latent_vector_logits, shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, common.LATENT_COUNT])

    # run controller
    hidden_output, hidden_internal = state
    controller_network_logits, hidden_output, hidden_internal = controller.network.inference(reshaped_zs + latent_noise, hidden_output, hidden_internal)
    state = hidden_output, hidden_internal
    reshaped_motor_commands = tf.reshape(controller_network_logits, shape=[controller.network.CONTROLLER_N * controller.network.LOCATION_N,
                                                                           controller.network.OUTPUT_DEPTH])

    # run movement sim
    local_changes = movement_sim.network.inference(reshaped_motor_commands)
    local_x_change, local_y_change, local_bearing_change = tf.unstack(local_changes, axis=1)
    local_bearing_change = local_bearing_change * 180.0

    # math to change relative movements to global movements
    distance_moved = tf.sqrt(tf.square(local_x_change) + tf.square(local_y_change))
    bearing_c = ((-tf.math.atan2(local_y_change, local_x_change) + (math.pi / 2)) % (math.pi * 2))

    temp_bearing_c = (tf.reshape(global_orientation, shape=[controller.network.CONTROLLER_N * controller.network.LOCATION_N]) / 180 * math.pi) + bearing_c
    global_x_change = distance_moved * tf.sin(temp_bearing_c)
    global_y_change = distance_moved * tf.cos(temp_bearing_c)
    local_bearing_change = tf.reshape(local_bearing_change, [controller.network.CONTROLLER_N, controller.network.LOCATION_N])
    global_x_change = tf.reshape(global_x_change, [controller.network.CONTROLLER_N, controller.network.LOCATION_N])
    global_y_change = tf.reshape(global_y_change, [controller.network.CONTROLLER_N, controller.network.LOCATION_N])

    new_global_orientation = (global_orientation + local_bearing_change) % 360
    new_global_position = tf.stack([global_x_change, global_y_change], axis=2) + global_position

    return new_global_position, new_global_orientation, state, controller_network_logits, reshaped_zs
