import sys
sys.path.append(__file__ + '/../..')
import numpy as np
import time
import random
import threading
import json
import os

import common
import robot.control
import robot.destination
import robot.tracker

if __name__ == '__main__':
    p1 = threading.Thread(target=robot.tracker.tracking_loop, args=(True,))
    p1.start()
    robot.control.open_port()
    print('Waiting for camera boots...')
    time.sleep(4)
    FILE_PATH = '../../data/raw/movement/%s.json' % time.strftime("%Y%m%d")
    common.mkdir_forfile(FILE_PATH)

    if os.path.isfile(FILE_PATH):
        print('Loaded old data.')
        with open(FILE_PATH, 'r') as fp:
            data = json.load(fp)
    else:
        data = {}

    def save():
        with open(FILE_PATH, 'w') as fp:
            json.dump(data, fp)

    step_id = len(data)
    print('Starting from step id: %d' % step_id)
    try:
        robot.destination.goto(np.array([0, 0]))
        print('Starting...')

        while True:
            left = random.uniform(-1, 1)
            right = random.uniform(-1, 1)

            old_center_x = robot.tracker.center[0]
            old_center_y = robot.tracker.center[1]
            old_bearing = robot.tracker.bearing

            robot.control.set_scaled_speed(left, right)
            time.sleep(common.ROBOT_MOTOR_STEP_TIME)  # move for this long
            robot.control.set_scaled_speed(0, 0)
            time.sleep(0.5)
            center_x = robot.tracker.center[0]
            center_y = robot.tracker.center[1]
            bearing = robot.tracker.bearing

            if abs(center_x) > common.BOARD_RANGE or abs(center_y) > common.BOARD_RANGE:
                print('Position reset.')
                robot.destination.goto(np.array([0, 0]))
                print('Reset done.')
                time.sleep(0.5)
            else:
                #good to save
                step = {}
                step['center_x'] = center_x
                step['center_y'] = center_y
                step['bearing'] = bearing
                step['old_center_x'] = old_center_x
                step['old_center_y'] = old_center_y
                step['old_bearing'] = old_bearing
                step['motor_left'] = left
                step['motor_right'] = right
                data['step_%d' % step_id] = step
                step_id += 1
                save()
                print('Saved step %d: %s' % (step_id, str(step)))

            print('Current position: %f, %f' % (robot.tracker.center[0], robot.tracker.center[1]))

    except KeyboardInterrupt:
        pass
    except ValueError:
        pass

    robot.control.close_port()
