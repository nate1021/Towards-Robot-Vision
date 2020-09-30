import sys
sys.path.append(__file__ + '/../..')
import threading
import math
import time
import numpy as np

import robot.tracker
import robot.control
import robot.onboard_cam


BEARING_ERROR_ALLOWANCE = 10  # DEGREES
CENTER_ERROR_ALLOWANCE = 0.025  # on -1 to 1 grid
TARGET_PROXIMITY_ALLOWANCE = 0.17  # on -1 to 1 grid
SAFE_ZONE = 0.9  # MAX distance from edges
RESET_SPEED = 0.5
TURN_EXTRA_SLOW = 0.8


def too_close_to_trackers(pos):
    for name in robot.tracker.trackers:
        if np.linalg.norm(pos - robot.tracker.trackers[name]) <= TARGET_PROXIMITY_ALLOWANCE:
            return True
    return False


def goto(target_xy, detour_inversion=False, spaces=''):
    print(spaces + 'Moving to: ' + str(target_xy))
    while np.linalg.norm(robot.tracker.center - target_xy) > CENTER_ERROR_ALLOWANCE:

        def turn_to_bearing(desired_bearing):
            while True:
                bearing_change = (desired_bearing - robot.tracker.bearing)  # positive = turn clockwise (bearing direction), negative = counterclockwise

                if bearing_change < -180:
                    bearing_change += 360
                elif bearing_change > 180:
                    bearing_change -= 360

                if abs(bearing_change) <= BEARING_ERROR_ALLOWANCE:
                    break  # done turning. can move towards center
                elif bearing_change > 0:
                    # turn clockwise=
                    robot.control.set_scaled_speed_classic(1 * RESET_SPEED * TURN_EXTRA_SLOW, -1 * RESET_SPEED * TURN_EXTRA_SLOW)
                elif bearing_change < 0:
                    # turn counterclockwise-
                    robot.control.set_scaled_speed_classic(-1 * RESET_SPEED * TURN_EXTRA_SLOW, 1 * RESET_SPEED * TURN_EXTRA_SLOW)

                time.sleep(0.1)  # 100MS per update

        # face target (previously called center)
        bearing_from_center = ((-math.atan2(robot.tracker.center[1] - target_xy[1], robot.tracker.center[0] - target_xy[0]) + math.pi / 2)
                               % (math.pi * 2)) / math.pi * 180
        desired_bearing = (bearing_from_center + 180) % 360
        turn_to_bearing(desired_bearing)
        # move towards center
        robot.control.set_scaled_speed_classic(1 * RESET_SPEED, 1 * RESET_SPEED)
        time.sleep(0.1)

        if too_close_to_trackers(robot.tracker.center):
            print(spaces + 'Too close to a tracker. Running detour...')
            robot.control.set_scaled_speed_classic(-1 * RESET_SPEED, -1 * RESET_SPEED)
            time.sleep(0.4)
            robot.control.set_scaled_speed_classic(0, 0)
            temp_target_bearing = (robot.tracker.bearing + 90) % 360

            detour_distance = 0
            detour_pos = None
            while detour_pos is None or too_close_to_trackers(detour_pos):
                detour_distance += -0.2 if detour_inversion else 0.2
                detour_pos = robot.tracker.center + np.array([detour_distance * math.sin(temp_target_bearing / 180 * math.pi),
                                                              detour_distance * math.cos(temp_target_bearing / 180 * math.pi)])
            out_of_bounds = abs(detour_pos[0]) > 0.9 or abs(detour_pos[1]) > 0.9
            if out_of_bounds:
                if len(spaces) == 0:  # top layer. Very messy code, should be fixed.
                    return goto(target_xy, not detour_inversion, spaces)
                else:
                    return False  # cannot navigate like that

            print(spaces + 'Detour distance: %f' % detour_distance)
            print(spaces + 'Detour target: %s' % str(detour_pos))

            success = goto(detour_pos, detour_inversion, spaces + '    ')
            if not success:
                print(spaces + 'Inverting detour route and starting again from start.')
                return goto(target_xy, not detour_inversion, spaces)

    robot.control.set_scaled_speed_classic(0, 0)
    print(spaces + 'Moved to: ' + str(target_xy))
    return True


if __name__ == '__main__':
    p1 = threading.Thread(target=robot.tracker.tracking_loop, args=(True,))
    p1.start()
    print('Waiting for camera boots...')
    time.sleep(4)  # wait for camera to start

    try:
        while True:
            robot.control.open_port()
            t_x = float(input('Target x: '))
            t_y = float(input('Target y: '))

            goto(np.array([t_x, t_y]))

    except KeyboardInterrupt:
        pass
    except ValueError:
        pass

    robot.control.close_port()
