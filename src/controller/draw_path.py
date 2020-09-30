import sys
sys.path.append(__file__ + '/../..')
import numpy as np
import cv2
from datetime import datetime
import math
from PIL import Image

import robot.tracker
import common

save_path = None
fig_stacked_images = []
backgrounds = None
image_count = 0


def cord_to_image(cord):
    if use_real_cam:
        return robot.tracker.cord_to_image(cord).astype(np.int)
    else:
        return (robot.tracker.cord_to_image(cord) - np.array([robot.tracker.StartX, robot.tracker.StartY])).astype(np.int)


use_real_cam = False


def clear(image_count_in, using_real_cam_arg, pos=0):
    global fig_stacked_images, backgrounds, save_path, image_count, use_real_cam
    image_count = image_count_in
    datestr = datetime.utcnow().strftime('%Y%m%d_%HH%MM%SS.%f')[:-3]

    if using_real_cam_arg:
        save_path = '../../output/experiment/real/%s/' % datestr
        backgrounds = np.ones(shape=[image_count_in,480,640,3]) * 200
    else:
        save_path = '../../output/controller/virtual/%d/' % pos
        backgrounds = np.zeros(shape=[image_count_in,robot.tracker.pixel_height,robot.tracker.pixel_width,3])
        backgrounds[:] = ((backgrounds[:] * 0.5) + 0.5) * 255

    common.mkdir(save_path)

    use_real_cam = using_real_cam_arg
    fig_stacked_images = []
    for i in range(image_count):
        fig_stacked_images.append([])


def draw_trackers(trackers):
    global backgrounds
    for i in range(image_count):
        cv2.rectangle(backgrounds[i], tuple(cord_to_image(trackers[common.TARGET_STRING]) - np.array([10, 10])),
                                                       tuple(cord_to_image(trackers[common.TARGET_STRING]) + np.array([10, 10])), common.TARGET_COLOR, 1)
        cv2.rectangle(backgrounds[i], tuple(cord_to_image(trackers[common.ANTITARGET_STRING]) - np.array([10, 10])),
                                                       tuple(cord_to_image(trackers[common.ANTITARGET_STRING]) + np.array([10, 10])), common.ANTITARGET_COLOR, 1)

        cv2.circle(backgrounds[i], tuple(cord_to_image(trackers[common.TARGET_STRING])),
                   int(common.TARGET_DESIRED_DISTANCE_TRAINING * robot.tracker.pixel_width / 2.0), common.TARGET_COLOR, 1)
        cv2.circle(backgrounds[i], tuple(cord_to_image(trackers[common.ANTITARGET_STRING])),
                   int(common.ANTI_TARGET_MIN_DISTANCE * robot.tracker.pixel_width / 2.0), common.ANTITARGET_COLOR, 1)


def draw_start_circles(centers, bearings, colors):
    global backgrounds
    for i in range(image_count):
        if colors is None:
            color = (150, 0, 150)
        else:
            color = colors[i]

        cv2.circle(backgrounds[i], tuple(cord_to_image(centers[i]).astype(np.int)), 10, color, 1)  # target
        cv2.line(backgrounds[i], tuple(cord_to_image(centers[i]).astype(np.int)),
                 tuple(cord_to_image(centers[i] + np.array([0.07 * math.sin(bearings[i] / 180 * math.pi),
                                                            0.07 * math.cos(bearings[i] / 180 * math.pi)])).astype(np.int)), color, 1)


def draw_lines(centers, old_centers, bearings, old_bearings, color):
    global backgrounds
    for i in range(image_count):
        cv2.line(backgrounds[i], tuple(cord_to_image(old_centers[i]).astype(np.int)),
                 tuple(cord_to_image(centers[i]).astype(np.int)), color, 1)
        if abs(old_bearings[i] - bearings[i]) >= 30 and np.linalg.norm(old_centers[i] - centers[i]) < 0.08:
            cv2.line(backgrounds[i], tuple(cord_to_image(centers[i]).astype(np.int)),
                     tuple(cord_to_image(centers[i] + np.array([0.03 * math.sin(bearings[i] / 180 * math.pi),
                                                                0.03 * math.cos(bearings[i] / 180 * math.pi)])).astype(np.int)), color, 1)


def save():
    for i in range(backgrounds.shape[0]):
        cv2.imwrite(save_path + 'tracking_%d.png' % i, backgrounds[i])
        if len(fig_stacked_images) > 0 and len(fig_stacked_images[i]) > 0:
            Image.fromarray(np.vstack(fig_stacked_images[i])).save(save_path + 'camera_%d.png' % i)
    print('Saved to %s.' % save_path)
