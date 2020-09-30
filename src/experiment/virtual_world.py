import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
import cv2
import math
from PIL import Image
import random
import threading

import common
import data_collection.camera_meta

BOARD_SIZE = 600
CAM_WIDTH = 600
CAM_HEIGHT = int(CAM_WIDTH / 4 * 3)
current_sim = 0
camera_image = None
zs_value = None
next_center = None
next_bearing = None
next_state_value = None
state_value = None


def worker(trackers_arr):
    global next_center, next_bearing, zs_value, camera_image, next_state_value, state_value
    import frame_compressor.network
    import frame_compressor.data
    import controller.network
    import controller.network_linker
    global_position_in = tf.placeholder(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2], dtype=tf.float32)  # global x, global y
    global_orientation_in = tf.placeholder(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N], dtype=tf.float32)  # global bearing
    state_in = controller.network.first_hidden_state_tensor()  # using this thing as a lazy placeholder
    global_position_out, global_orientation_out, state_out, controller_network_logits_out, zs = \
        controller.network_linker.next_position_orientation(global_position_in, global_orientation_in, state_in, 0)
    zs = tf.reshape(zs, shape=[controller.network.CONTROLLER_N * controller.network.LOCATION_N, common.LATENT_COUNT])
    zs = tf.unstack(zs)[0]
    zs = tf.reshape(zs, shape=[1, common.LATENT_COUNT])
    _, _, _, final_image = frame_compressor.network.encoder(tf.zeros([1, common.IMAGE_HEIGHT, common.IMAGE_WIDTH, common.IMAGE_DEPTH]))
    reconstruct = frame_compressor.network.decoder(zs, final_image)
    sim_frame_compressor_reconstruct_post = frame_compressor.data.reverse_parse(reconstruct)

    population = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.gene_count])
    try:
        population[0] = np.loadtxt('../../saved_models/best_controller.csv')
    except IOError:
        print('WARNING! No controller loaded!')

    gene_feed_dict = controller.network.gen_feed_dict(population)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        controller.network_linker.restore_models(trackers_arr, sess)
        _, restore = frame_compressor.network.get_saver()
        restore(sess)

        old_center = np.array([0, 0])
        old_bearing = 0
        old_current_sim = -1
        while True:
            # run networks
            sim_change = False
            if current_sim != old_current_sim:
                old_current_sim = current_sim
                sim_change = True
                controller.network_linker.load_latent_simulator(current_sim, sess)

            if sim_change or old_center[0] != center[0] or old_center[1] != center[1] or old_bearing != bearing:
                old_center = np.array(center)
                old_bearing = bearing

                global_position = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2])  # global x, global y
                global_orientation = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N])  # global bearing
                global_position[0, 0, :] = center
                global_orientation[0, 0] = bearing

                if state_value is None:
                    state_value = controller.network.first_hidden_state_numpy()
                for ii in range(len(state_in)):
                    for jj in range(len(state_in[ii])):
                        gene_feed_dict[state_in[ii][jj]] = state_value[ii][jj]
                gene_feed_dict[global_position_in] = global_position
                gene_feed_dict[global_orientation_in] = global_orientation
                global_position_out_value, global_orientation_out_value, next_state_value, motor_commands, zs_value, recs = \
                    sess.run((global_position_out, global_orientation_out, state_out, controller_network_logits_out, zs, sim_frame_compressor_reconstruct_post),
                             feed_dict=gene_feed_dict)

                next_center = global_position_out_value[0, 0, :]
                next_bearing = global_orientation_out_value[0, 0]
                motor_commands = motor_commands[0, 0, :]

                camera_image_small_pil = Image.fromarray(recs[0].astype(np.uint8))
                pil_img_big = camera_image_small_pil.resize((CAM_WIDTH, CAM_HEIGHT))
                camera_image = cv2.cvtColor(np.array(pil_img_big), cv2.COLOR_RGB2BGR) / 255.0
            # done running networks


if __name__ == '__main__':
    trackers_arr = data_collection.camera_meta.load_trackers()
    p1 = threading.Thread(target=worker, args=(trackers_arr,))
    p1.start()

    window_title = "Virtual World"

    cv2.namedWindow(window_title)

    def cord_to_image(cord_x, cord_y):
        pixel_x = (cord_x + 1) * (BOARD_SIZE / 2)
        pixel_y = ((-1 * cord_y) + 1) * (BOARD_SIZE / 2)
        return int(pixel_x), int(pixel_y)

    center = np.array([0, 0])
    bearing = 45
    save_count = 0

    current_tracker = trackers_arr[0]
    OUT_PATH = '../../output/virtual_world/'
    common.mkdir(OUT_PATH)
    while True:
        img = np.zeros(shape=[BOARD_SIZE,BOARD_SIZE + CAM_WIDTH,3])  # H,W,C with C is BGR
        img[:,0:BOARD_SIZE,:] += 0.95

        for ky in current_tracker.keys():
            if ky != 'position':
                item = current_tracker[ky]
                if ky == common.TARGET_STRING:
                    clr = (common.TARGET_COLOR[0] / 255.0, common.TARGET_COLOR[1] / 255.0, common.TARGET_COLOR[2] / 255.0)
                elif ky == common.ANTITARGET_STRING:
                    clr = (common.ANTITARGET_COLOR[0] / 255.0, common.ANTITARGET_COLOR[1] / 255.0, common.ANTITARGET_COLOR[2] / 255.0)
                cv2.circle(img, cord_to_image(item[0], item[1]), 12, clr, -1)  # target

        cv2.line(img, cord_to_image(-1, 0), cord_to_image(1, 0), (0.7, 0.7, 0.7), 1)
        cv2.line(img, cord_to_image(0, -1), cord_to_image(0, 1), (0.7, 0.7, 0.7), 1)

        cv2.circle(img, cord_to_image(center[0], center[1]), 23, (0,0,0), 2)  # target

        cv2.putText(img, ("x:%.3f, y:%.3f, bearing: %0.f" % (center[0], center[1], bearing)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        ARROW_LENGTH = 0.08
        cv2.line(img, cord_to_image(center[0], center[1]), cord_to_image(center[0] + (ARROW_LENGTH * math.sin(bearing / 180 * math.pi)),
                                                                         center[1] + (ARROW_LENGTH * math.cos(bearing / 180 * math.pi))), (0, 0, 0), 2)

        if camera_image is not None:
            for i in range(common.LATENT_COUNT):
                cv2.putText(img, "z[%d]: %.4f" % (i, zs_value[0, i]), (5, 10 + ((i + 1) * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

            img[0:CAM_HEIGHT, BOARD_SIZE:BOARD_SIZE + CAM_WIDTH, :] = camera_image
        cv2.imshow(window_title, img)

        BEARING_SPEED = 4
        MOVEMENT_SPEED = 0.007
        ky = cv2.waitKey(10) & 0xFF
        if ky == ord('q'):
            cv2.destroyAllWindows()
            break
        elif ky == ord('a'):
            state_value = None
            bearing -= BEARING_SPEED
            bearing %= 360
        elif ky == ord('d'):
            state_value = None
            bearing += BEARING_SPEED
            bearing %= 360
        elif ky == ord('w'):
            state_value = None
            change = np.array([math.sin(bearing / 180 * math.pi), math.cos(bearing / 180 * math.pi)])
            center = center + (change * MOVEMENT_SPEED)
        elif ky == ord('s'):
            state_value = None
            change = np.array([math.sin(bearing / 180 * math.pi), math.cos(bearing / 180 * math.pi)])
            center = center - (change * MOVEMENT_SPEED)
        elif ky == ord('r'):
            print('Using controller for 1 step...')
            print('State hash before: %f' % (np.sum(np.array(state_value))))
            center = next_center
            bearing = next_bearing
            state_value = next_state_value
        elif ky == ord('f'):
            state_value = None
            center = np.random.uniform(low=-common.BOARD_RANGE, high=common.BOARD_RANGE, size=[2])
            bearing = random.uniform(0, 360)
        elif ky == ord('e'):
            cv2.imwrite(OUT_PATH + '%d.png' % save_count, camera_image * 255.0)
            save_count += 1
            print('saved')

        for k in range(1,7):
            if ky == ord(str(k)):
                current_sim = k - 1
                current_tracker = trackers_arr[k - 1]
                state_value = None
