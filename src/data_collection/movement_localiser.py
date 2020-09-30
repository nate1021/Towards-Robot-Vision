import sys
sys.path.append(__file__ + '/../..')
import json
import math
import numpy as np

import common

FILE_PATH = '../../data/raw/movement/20190318.json'
OUT_PATH = '../../data/processed/localised_movements.csv'
common.mkdir_forfile(OUT_PATH)
with open(FILE_PATH, 'r') as fp:
    measurements = json.load(fp)

print('Total steps: %d' % len(measurements))
bad_steps = []
for i in range(1300, 1310):
    bad_steps.append(i)
for i in range(2590, 2602):
    bad_steps.append(i)
for i in range(3240, 3245):
    bad_steps.append(i)
for i in range(4535, 4538):
    bad_steps.append(i)

print('Not including steps: %s' % bad_steps)

outp = []

for i in range(len(measurements)):
    if i not in bad_steps:
        step = measurements['step_%d' % i]
        center_x = step['center_x']
        center_y = step['center_y']
        bearing = step['bearing']  # anything, doesnt matter
        old_center_x = step['old_center_x']
        old_center_y = step['old_center_y']
        old_bearing = step['old_bearing']

        bearingB = ((-math.atan2(center_y - old_center_y, center_x - old_center_x) + math.pi / 2) % (math.pi * 2)) \
                    / math.pi * 180
        tempBearingB = bearingB - old_bearing
        distanceB = np.linalg.norm(np.array([center_x, center_y]) - np.array([old_center_x, old_center_y]))
        localXChange = distanceB * math.sin(tempBearingB / 180 * math.pi)
        localYChange = distanceB * math.cos(tempBearingB / 180 * math.pi)
        localBearingChange = bearing - old_bearing
        if localBearingChange < -180:
            localBearingChange += 360
        if localBearingChange > 180:
            localBearingChange -= 360
        print('%d: localXChange: %.4f, localYChange: %.4f, localBearingChange: %.2f, left: %.3f, right: %.3f' %
              (len(outp), localXChange, localYChange, localBearingChange, step['motor_left'], step['motor_right']))
        outp.append([step['motor_left'], step['motor_right'], localXChange, localYChange, localBearingChange])

outp = np.array(outp)
np.savetxt(OUT_PATH, outp)
print('Saved to: ' + OUT_PATH)
