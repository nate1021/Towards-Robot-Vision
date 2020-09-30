import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
from PIL import Image
from stopwatch import Stopwatch

import common
import frame_compressor.network
import frame_compressor.data

dataset_init_op, dataset_full_init_op, validation_placeholder, position, images, filenames = frame_compressor.data.datasets()

z, mean, log_std, final_image = frame_compressor.network.encoder(images)
reconstruct = frame_compressor.network.decoder(z, final_image)

save, restore = frame_compressor.network.get_saver()

KLD_loss_placeholder = tf.placeholder(dtype=tf.float32)
L2_loss = tf.reduce_sum(tf.squared_difference(images, reconstruct), axis=[1, 2, 3])
L2_loss = tf.reduce_mean(L2_loss)
_std = tf.exp(log_std)
KLD_loss = 0.5 * tf.reduce_sum(tf.multiply(_std, _std) + tf.multiply(mean, mean) - log_std - 1, axis=1)
KLD_loss = tf.maximum(KLD_loss, 0.5 * common.LATENT_COUNT)  # from worldModels paper
KLD_loss = tf.reduce_mean(KLD_loss)
loss = L2_loss + KLD_loss

output_real = frame_compressor.data.reverse_parse(images)
output_reconstruct = frame_compressor.data.reverse_parse(reconstruct)


def run_tests(sess):
    outpath = '../../output/frame_compressor'
    common.mkdir(outpath)
    sess.run(dataset_full_init_op, feed_dict={validation_placeholder: True})
    a, b, zs, lss, kld_lss, l2_lss, position_out = sess.run((output_real, output_reconstruct, z, loss, KLD_loss, L2_loss, position))
    print('Validation Loss: %4f, (KLD: %4f, L2: %4f)' % (lss, kld_lss, l2_lss))
    print('Validation positions: %s' % str(position_out))
    hori = 10
    poses = list(set(position_out))
    buffer = np.zeros(shape=[a[0].shape[0], 10, 3])

    for p in poses:
        a_sub = []
        b_sub = []
        for i in range(len(a)):
            if position_out[i] == p:
                a_sub.append(a[i])
                b_sub.append(b[i])

        combo = []
        for i in range(0, len(a_sub) - (hori - 1), hori):
            lst = []
            for k in range(i, i + hori):
                lst.append(a_sub[k])
                lst.append(b_sub[k])
                lst.append(buffer)
            combo.append(np.hstack(lst))

        Image.fromarray(np.vstack(combo).astype(np.uint8)).save((outpath + '/test_pos%s.png') % p)
    return lss


if __name__ == '__main__':
    # turning down the Beta's reduces chance of been stuck in average
    train = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.8, beta2=0.9, epsilon=1e-10).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        best_lss = 1e10
        step = 0

        tests_without_improvement = 0
        stopwatch = Stopwatch()
        while stopwatch.duration <= 45 * 60:
            sess.run(dataset_init_op, feed_dict={validation_placeholder: False})

            for i in range(1000):
                _, lss, kld_lss, l2_lss, positions = sess.run((train, loss, KLD_loss, L2_loss, position))
                if i % 10 == 0:
                    print('%d  Loss: %4f, KLD: %4f, L2: %4f' % (step, lss, kld_lss, l2_lss))

                step += 1
            print('Training positions: %s' % str(positions))

            lss = run_tests(sess)

            if lss < best_lss:
                best_lss = lss
                tests_without_improvement = 0
                save(sess)
                print('Saved.')
            else:
                tests_without_improvement += 1
                print('Tests without improvement: %d' % tests_without_improvement)

            print('Trained for %d mins' % int(stopwatch.duration / 60.0))
