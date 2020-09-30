import numpy as np
import json
import cv2
import os
import robot_tracker
import colorsys

BOARD_SIZE = 600

pos = 1
while True:
    data_name = 'real_export__pos%d/' % pos # input('Enter real_export folder name:')

    if not os.path.exists(data_name):
        break

    with open('%sdata.json' % data_name, 'r') as fp:
        data = json.load(fp)

    base = np.ones([480,640,3]) #H,W,C

    #PAST ROBOTS
    # for ri in range(30):
    #     run = data['run_%d' % ri]
    #     img = cv2.imread(data_name + '/' + run['location_0']['path_overhead'].split('/')[-1]) / 255.0
    #     if ri == 0:
    #         base[:] = img[:]
    #     else:
    #         px, py = robot_tracker.cord_to_image((float(run['location_0']['center_x']), float(run['location_0']['center_y'])))
    #         ROBOT_SIZE_SQUARE = 50
    #         ROBOT_SIZE_CIRCLE = 15
    #         px_min = int(px - (ROBOT_SIZE_SQUARE / 2))
    #         py_min = int(py - (ROBOT_SIZE_SQUARE / 2))
    #         #base[py:py+ROBOT_SIZE,px:px+ROBOT_SIZE,:] = img[py:py+ROBOT_SIZE,px:px+ROBOT_SIZE,:]
    #         for y in range(py_min, py_min + ROBOT_SIZE_SQUARE):
    #             for x in range(px_min, px_min + ROBOT_SIZE_SQUARE):
    #                 distance = np.linalg.norm(np.array([x,y] - np.array([px,py])))
    #                 if distance < ROBOT_SIZE_CIRCLE:
    #                     base[y, x, :] = img[y, x, :]



    def get_image(run_index, location_index, path_end):
        return cv2.imread(data_name + data['run_%d' % run_index]['location_%d' % location_index]['path_' + path_end].split('/')[-1]) / 255.0

    def get_center(run_index, location_index):
        return np.array([float(data['run_%d' % run_index]['location_%d' % location_index]['center_x']), float(data['run_%d' % run_index]['location_%d' % location_index]['center_y'])])

    #build clean background
    base[:] = get_image(0, 0, 'overhead')
    first_center = get_center(0,0)
    run_index = 1
    while True:
        this_center = get_center(run_index,0)
        distance2 = np.linalg.norm(first_center - this_center)
        if distance2 > 0.5:

            ROBOT_SIZE_SQUARE = 100
            px, py = robot_tracker.cord_to_image(first_center)
            px = int(px - (ROBOT_SIZE_SQUARE / 2))
            py = int(py - (ROBOT_SIZE_SQUARE / 2))
            base[py:py+ROBOT_SIZE_SQUARE,px:px+ROBOT_SIZE_SQUARE,:] = get_image(run_index, 0, 'overhead')[py:py+ROBOT_SIZE_SQUARE,px:px+ROBOT_SIZE_SQUARE,:]

            break
        run_index += 1

    #Double draw size for cleaner lines
    UPSCALE = 2
    base = cv2.resize(base, (0,0), fx=UPSCALE, fy=UPSCALE, interpolation = cv2.INTER_CUBIC)
    robot_tracker.StartX *= UPSCALE
    robot_tracker.StartY *= UPSCALE
    robot_tracker.pixel_width *= UPSCALE
    robot_tracker.pixel_height *= UPSCALE

    #Draw lines
    def get_hue(dev_h_value, dev_v_value):
        r, g, b = colorsys.hsv_to_rgb(dev_h_value, 1, dev_v_value)
        return (b, g, r)

    def cord_to_image(coord):
        return  tuple(robot_tracker.cord_to_image(coord).astype(np.int32))

    PROGRESSIVE_LINE_MODE = True
    for run_index in range(30):
        run = data['run_%d' % run_index]
        location_total = 0
        old_position = None
        while 'location_%d' % location_total in run:
            location_total += 1
        for location_i in range(location_total):
            new_position = get_center(run_index, location_i)
            if PROGRESSIVE_LINE_MODE:
                clr = get_hue((location_total - location_i) / float(location_total) * 0.7, 1)
            else:
                clr = get_hue(run_index / 30.0, 1 if run_index % 2 == 0 else 0.75)

            if location_i > 0:
                cv2.line(base, cord_to_image(old_position), cord_to_image(new_position), clr, 2)

            old_position = new_position

    base = cv2.resize(base, (0,0), fx=1.0/UPSCALE, fy=1.0/UPSCALE, interpolation = cv2.INTER_CUBIC)
    base = base[0:420,130:575]
    robot_tracker.StartX /= UPSCALE
    robot_tracker.StartY /= UPSCALE
    robot_tracker.pixel_width /= UPSCALE
    robot_tracker.pixel_height /= UPSCALE


    #reconstruct section


    path = '_movement_map_%d.png' % pos
    cv2.imwrite(path, base * 255)
    print('Saved to %s' % path)
    pos += 1

    # window_title = "Window"
    # while True:
    #     cv2.namedWindow(window_title)
    #     cv2.imshow(window_title, base)
    #     ky = cv2.waitKey(10) & 0xFF
    #     if ky == ord('n'):
    #         break
    #
    # cv2.destroyAllWindows()
