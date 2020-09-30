import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import os
from PIL import Image
import json

import common

DATA_PATH = '../../data/raw/camera'
META_PATH = DATA_PATH + '/meta'
OUTPUT_FILE_DIR = '../../data/processed/camera'
common.mkdir(OUTPUT_FILE_DIR)
OUTPUT_FILE = OUTPUT_FILE_DIR + '/images.tfrecord'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


counter = 0

print('Generating %s' % OUTPUT_FILE)
with tf.python_io.TFRecordWriter(OUTPUT_FILE) as record_writer:

    ifile = 1
    while True:
        filepath = META_PATH + '/position%d.json' % ifile
        if not os.path.exists(filepath):
            break

        with open(filepath, 'r') as fp:
            location_data = json.load(fp)

        out_data = []
        for step_key in location_data:
            if step_key.startswith('step_'):
                image_name = DATA_PATH + '/' + location_data[step_key]['onboard_image']
                if not os.path.exists(image_name):
                    print('No image found for: ' + step_key)
                else:
                    print('Wrote: %d\t%s in %s' % (counter, image_name, filepath))
                    image = Image.open(image_name)
                    arr = image_name.split('/')[-1].split('.png')[0]
                    image = image.resize((common.IMAGE_WIDTH, common.IMAGE_HEIGHT), Image.ANTIALIAS)
                    imgb = image.convert('RGB').tobytes()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'input': _bytes_feature(imgb),
                            'filename': _bytes_feature(arr.encode()),
                            'position': _int64_feature(ifile),
                            'index': _int64_feature(counter)
                        }))
                    counter += 1
                    record_writer.write(example.SerializeToString())

        ifile += 1

print('Done.')
