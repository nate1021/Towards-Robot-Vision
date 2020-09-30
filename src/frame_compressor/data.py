import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf

import common

SOURCE_PATH = '../../data/processed/camera/images.tfrecord'
TESTING_PERCENT = 10


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'input': tf.FixedLenFeature([], tf.string),
            'filename': tf.FixedLenFeature([], tf.string),
            'position': tf.FixedLenFeature([], tf.int64),
            'index': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['input'], tf.uint8)
    filename = features['filename']
    position = features['position']
    index = features['index']

    image = tf.reshape(image, shape=[common.IMAGE_HEIGHT, common.IMAGE_WIDTH, common.IMAGE_DEPTH])
    image = ((tf.cast(image, tf.float32) / 255.0) * 2) - 1  # ranging between -1 and 1

    return index, position, image, filename


validation_placeholder = tf.placeholder(dtype=tf.bool)


def filter(index, position, *useless_args):
    # index_mod100 = index % 100
    # condition = index_mod100 < TESTING_PERCENT
    condition = tf.logical_and(position >= 7, position <= 9) #True for test
    return tf.where(validation_placeholder, condition, tf.logical_not(condition))


def reverse_parse(images):
    return tf.cast(tf.maximum(0.0, tf.minimum(255.0, ((images + 1) / 2) * 255.0)), tf.uint8)


def datasets():
    max_batch_size = 64
    dataset = tf.data.TFRecordDataset([SOURCE_PATH])
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.map(parser)
    dataset = dataset.filter(filter)
    dataset_full = dataset
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=max_batch_size)
    dataset_full = dataset_full.batch(batch_size=1024)

    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset)
    dataset_full_init_op = iterator.make_initializer(dataset_full)

    _, position, images, filenames = iterator.get_next()

    return dataset_init_op, dataset_full_init_op, validation_placeholder, position, images, filenames
