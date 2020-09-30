import sys
sys.path.append(__file__ + '/../..')
import json
import numpy as np
from datetime import datetime
import time
import random
import threading
import cv2
import os

import common
import robot.tracker
import robot.control
import robot.onboard_cam
import robot.destination

IMAGES_PER_POINT = 12

DATA_PATH = '../../data/raw/camera'
META_PATH = DATA_PATH + '/meta'
common.mkdir(META_PATH)
common.mkdir(DATA_PATH + '/images')


if __name__ == '__main__':
    COLOR_DONE_THIS_SESSION = (255, 0, 255)
    COLOR_CURRENT = (255, 255, 0)
    COLOR_DONE_LAST_SESSION = (0, 185, 255)
    COLOR_TOO_CLOSE = (0, 0, 255)
    COLOR_TO_DO = (100, 100, 100)
    robot.tracker.data_gen_point_list[COLOR_CURRENT] = []
    robot.tracker.data_gen_point_list[COLOR_DONE_THIS_SESSION] = []
    robot.tracker.data_gen_point_list[COLOR_DONE_LAST_SESSION] = []
    robot.tracker.data_gen_point_list[COLOR_TOO_CLOSE] = []
    robot.tracker.data_gen_point_list[COLOR_TO_DO] = []


    command = str(input('Random uniform (r) or grid (g)?\n')).lower()
    if command == 'r':
        command = str(input('Generate new point map? (y/n)\n')).lower()
        if command == 'y':
            command = str(input('Are you sure? (y/n)\n')).lower()

        POINTMAP_PATH = META_PATH + '/pointmap.csv'

        if command == 'y':
            points_gen = np.random.uniform(size=[8,2],low=-common.BOARD_RANGE,high=common.BOARD_RANGE)
            points = np.zeros_like(points_gen)
            points_gen = list(points_gen)
            countup = 0
            cur = points_gen[0]
            del points_gen[0]
            points[countup] = cur
            countup += 1
            while len(points_gen) > 0:
                best_match = 0
                for i in range(len(points_gen)):
                    if np.linalg.norm(points_gen[best_match] - cur) > np.linalg.norm(points_gen[i] - cur):
                        best_match = i

                cur = points_gen[best_match]
                del points_gen[best_match]
                points[countup] = cur
                countup += 1

            if countup != points.shape[0]:
                input('FAILED UNIT TEST.')

            np.savetxt(POINTMAP_PATH, points)
        else:
            points = np.loadtxt(POINTMAP_PATH)
    else:
        points = []
        yy = -common.BOARD_RANGE
        while yy < common.BOARD_RANGE:
            xx = -common.BOARD_RANGE
            while xx < common.BOARD_RANGE:
                points.append([xx, yy])
                xx += 0.1
            yy += 0.1
        points = np.array(points)

    p2 = threading.Thread(target=robot.onboard_cam.camera_loop, args=())
    p2.start()
    p1 = threading.Thread(target=robot.tracker.tracking_loop, args=(True,))
    p1.start()
    robot.control.open_port()
    print('Waiting for camera boots...')
    time.sleep(4)

    timestamp = datetime.utcnow().strftime("%Y%m%d")
    META_DATA_FILE_PATH = '%s/%s.json' % (META_PATH, timestamp)
    cv2.imwrite('%s/%s.png' % (META_PATH, timestamp), robot.tracker.frame)

    if os.path.isfile(META_DATA_FILE_PATH):
        print('Loaded old data.')
        with open(META_DATA_FILE_PATH, 'r') as fp:
            data = json.load(fp)
    else:
        data = {}


    def save():
        with open(META_DATA_FILE_PATH, 'w') as fp:
            json.dump(data, fp)


    step_id = len(data)
    print('Step_id: %d' % step_id)
    try:
        trackers = {}
        for name in robot.tracker.trackers:
            trackers[name] = list(robot.tracker.trackers[name])

        data['trackers'] = trackers
        already_done = {}
        for k in data:
            if k.startswith('step_'):
                target = (data[k]['target_x'],data[k]['target_y'])
                if target not in already_done:
                    already_done[target] = 1
                else:
                    already_done[target] += 1

        save()

        for p in range(points.shape[0]):
            robot.tracker.data_gen_point_list[COLOR_TO_DO].append(tuple(points[p]))

        for p in range(points.shape[0]):
            xx = points[p,0]
            yy = points[p,1]
            to_remove = robot.tracker.data_gen_point_list[COLOR_TO_DO].index((xx,yy))
            del robot.tracker.data_gen_point_list[COLOR_TO_DO][to_remove]

            if robot.destination.too_close_to_trackers(np.array([xx, yy])):
                robot.tracker.data_gen_point_list[COLOR_TOO_CLOSE].append((xx, yy))
                print('Skipping grid point (%f, %f) (too close to red)' % (xx, yy))
            else:
                if (xx, yy) in already_done and already_done[(xx, yy)] >= IMAGES_PER_POINT:
                    robot.tracker.data_gen_point_list[COLOR_DONE_LAST_SESSION].append((xx, yy))
                    print('Already done: %f, %f.' % (xx, yy))
                else:
                    robot.tracker.data_gen_point_list[COLOR_CURRENT].append((xx, yy))
                    print('Desired target: %f, %f' % (xx, yy))
                    current_required = IMAGES_PER_POINT
                    if (xx, yy) in already_done:
                        current_required -= already_done[(xx, yy)]
                    print('Steps still needed: %d' % current_required)

                    robot.destination.goto(np.array([xx, yy]))
                    if random.uniform(0, 1) >= 0.5:
                        one = -1.0
                    else:
                        one = 1.0

                    robot.control.set_scaled_speed_classic(1 * one, -1 * one)
                    time.sleep(random.uniform(0.05, 1.5))
                    for i in range(current_required):
                        robot.control.set_scaled_speed_classic(1 * one, -1 * one)
                        time.sleep(random.uniform(0.05, 0.5))
                        robot.control.set_scaled_speed_classic(0, 0)

                        time.sleep(0.2)
                        step = {}
                        step['center_x'] = robot.tracker.center[0]
                        step['center_y'] = robot.tracker.center[1]
                        step['target_x'] = xx
                        step['target_y'] = yy
                        step['bearing'] = robot.tracker.bearing
                        id = datetime.utcnow().strftime('%Y%m%d_%HH%MM%SS.%f')[:-3]
                        fname = 'images/' + id + '.png'
                        cv2.imwrite(DATA_PATH + '/' + fname, robot.onboard_cam.frame)
                        step['onboard_image'] = fname
                        data['step_%d' % step_id] = step
                        step_id += 1
                        save()

                    robot.tracker.data_gen_point_list[COLOR_DONE_THIS_SESSION].append((xx, yy))
                    robot.tracker.data_gen_point_list[COLOR_CURRENT] = []
                    print('Collecting at: %f, %f\n' % (robot.tracker.center[0], robot.tracker.center[1]))

    except KeyboardInterrupt:
        pass
    except ValueError:
        pass

    robot.control.close_port()
