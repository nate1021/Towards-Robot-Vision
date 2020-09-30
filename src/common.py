import os

LATENT_COUNT = 8

IMAGE_DEPTH = 3
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48

BOARD_RANGE = 0.8  # workable board range. Max is 1 for [-1, 1]
ROBOT_MOTOR_STEP_TIME = 0.5  # run motors for 500ms

TARGET_STRING = 'red'
ANTITARGET_STRING = 'yellow'
TARGET_COLOR = (0, 0, 255)
ANTITARGET_COLOR = (0, 255, 255)

TARGET_DESIRED_DISTANCE_TRAINING = 0.12  # with reference to a [-1,1] board
ANTI_TARGET_MIN_DISTANCE = 0.4


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def mkdir_forfile(file):
    mkdir(os.path.dirname(file))
