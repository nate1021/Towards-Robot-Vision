import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import os
from stopwatch import Stopwatch

import latent_sim.network
import latent_sim.data


is_training = tf.placeholder(dtype=tf.bool)


dataset_test_init_ops = []
losses = []
dataset_train_init_ops = []
trains = []
randomise_maskses = []
saves = []
restores = []
var_lists = []
positions = []
best_lsses = []

ifile = 1
while True:
    filepath = latent_sim.data.SOURCE_TEMPLATE % ifile
    if os.path.exists(filepath):

        dataset_test_init_op, dataset_train_init_op, x_input, y_desired = latent_sim.data.datasets(ifile)

        blank = '%d_' % ifile
        y_logits, low_level_masks, var_list = latent_sim.network.inference(x_input, is_training, blank=blank)

        loss = tf.reduce_mean(tf.squared_difference(y_logits, y_desired))
        # turning down the Beta's reduces chance of been stuck in average
        train = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)

        randomise_masks = tf.variables_initializer(var_list=low_level_masks)

        save, restore = latent_sim.network.get_saver(ifile, blank=blank)

        dataset_test_init_ops.append(dataset_test_init_op)
        losses.append(loss)
        dataset_train_init_ops.append(dataset_train_init_op)
        trains.append(train)
        randomise_maskses.append(randomise_masks)
        saves.append(save)
        restores.append(restore)
        var_lists.append(var_list)
        positions.append(ifile)
        best_lsses.append(1e10)
    else:
        break
    ifile += 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    step = 0

    def run_test():
        sess.run(dataset_test_init_ops)

        losses_out = sess.run(losses, feed_dict={is_training: False})
        tmp = ''
        for l in losses_out:
            tmp += '%4f ' % l
        print('Test loss: %s' % tmp)

        return losses_out

    stopwatch = Stopwatch()
    while stopwatch.duration <= 45 * 60:

        sess.run(dataset_train_init_ops)
        for i in range(5000):
            _, losses_out, _ = sess.run((trains, losses, randomise_maskses), feed_dict={is_training: True})
            if i % 1000 == 0:
                tmp = ''
                for l in losses_out:
                    tmp += '%4f ' % l
                print('%d  Loss: %s' % (step, tmp))
            step += 1

        losses_out = run_test()
        for i in range(len(losses_out)):
            if losses_out[i] < best_lsses[i]:
                best_lsses[i] = losses_out[i]

                saves[i](sess)
                print('Saved %d with loss %4f' % (i, losses_out[i]))

        print('Trained for %d mins' % int(stopwatch.duration / 60.0))

    print()
    print('Training complete. Finialising models...')
    print()

    for index in range(len(positions)):
        print('Renaming model %d' % positions[index])
        restores[index](sess)
        print('Best state restored.')

        new_var_list = []
        for var in var_lists[index]:
            new_var_name = var.name.split('_')[1].split(':')[0]
            new_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(new_var)
            print('Old name: %s, New name: %s' % (var.name, new_var_name))

        saver = tf.train.Saver(var_list=new_var_list)
        path_model = latent_sim.network.SAVE_PATH_TEMPLATE % positions[index]
        sess.run(tf.variables_initializer(var_list=new_var_list))
        print('Sanity check - Old var name: %s, New var name %s, Old var value: %s, New var value %s' % (var.name, new_var.name, str(sess.run(var)), str(sess.run(new_var))))
        saver.save(sess=sess, save_path=path_model)
        print('Saved.')

    print('Done.')
