import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
from stopwatch import Stopwatch

import movement_sim.network
import movement_sim.data

dataset_train_init_op, dataset_test_init_op, x_input, y_desired = movement_sim.data.datasets()
y_logits = movement_sim.network.inference(x_input)

loss = tf.reduce_mean(tf.squared_difference(y_logits, y_desired))

# turning down the Beta's reduces chance of been stuck in average
train = tf.train.AdamOptimizer().minimize(loss)

save, restore = movement_sim.network.get_saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    best_lss = 1e10
    step = 0

    np.set_printoptions(suppress=True, precision=3)

    def run_test():
        sess.run(dataset_test_init_op)

        lss, y_logits_out, y_desired_out = sess.run((loss, y_logits, y_desired))
        for i in range(10):
            print('%s\tvs\t%s' % (y_desired_out[i], y_logits_out[i]))
        print('Test loss: %4f' % lss)

        return lss

    stopwatch = Stopwatch()
    while stopwatch.duration <= 45 * 60:

        sess.run(dataset_train_init_op)
        for i in range(5000):
            _, lss = sess.run((train, loss))
            if i % 1000 == 0:
                print('%d  Loss: %4f' % (step, lss))
            step += 1

        lss = run_test()

        if lss < best_lss:
            best_lss = lss
            save(sess)
            print('Saved.')

        print('Trained for %d mins' % int(stopwatch.duration / 60.0))
