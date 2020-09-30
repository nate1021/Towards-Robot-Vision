import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np


def datasets():
    raw_data = np.loadtxt('../../data/processed/localised_movements.csv')
    x_data = np.zeros(shape=[raw_data.shape[0], 2])
    x_data[:, 0] = raw_data[:, 0]  # left motor
    x_data[:, 1] = raw_data[:, 1]  # right motor

    y_data = np.zeros(shape=[raw_data.shape[0], 3])
    y_data[:, 0] = raw_data[:, 2]  # xchange
    y_data[:, 1] = raw_data[:, 3]  # ychange
    y_data[:, 2] = raw_data[:, 4] / 180.0  # bearing change

    test_index = int(x_data.shape[0] / 10)
    # if this doesnt work, limit controller speed between 0.8 and -0.8

    x_train = x_data[test_index:None]
    x_test = x_data[0:test_index]
    y_train = y_data[test_index:None]
    y_test = y_data[0:test_index]

    max_batch_size = 128
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=max_batch_size * 20)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.batch(batch_size=max_batch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset_test = dataset_test.batch(batch_size=x_test.shape[0])

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    dataset_train_init_op = iterator.make_initializer(dataset_train)
    dataset_test_init_op = iterator.make_initializer(dataset_test)

    x_input, y_desired = iterator.get_next()
    x_input = tf.cast(x_input, tf.float32)
    y_desired = tf.cast(y_desired, tf.float32)

    return dataset_train_init_op, dataset_test_init_op, x_input, y_desired
