import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
from random import shuffle
import os
import json

import common
import frame_compressor.network
import frame_compressor.data

dataset_init_op, dataset_full_init_op, validation_placeholder, position, images, filenames = frame_compressor.data.datasets()

z, mean, log_std, final_image = frame_compressor.network.encoder(images)

save, restore = frame_compressor.network.get_saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    restore(sess)
    print('Model restored.')

    sess.run(dataset_full_init_op, feed_dict={validation_placeholder: False})

    latent_data = {}
    count = 0
    while True:
        try:
            z_out, filenames_out = sess.run((z, filenames))
            for i in range(z_out.shape[0]):
                for j in range(z_out.shape[1]):
                    latent_data[filenames_out[i].decode()] = z_out[i]
                count += 1
            print('Processed %d latent vector.' % count)
        except tf.errors.OutOfRangeError:
            break

    ifile = 1
    while True:
        filepath = '../../data/raw/camera/meta/position%d.json' % ifile
        if os.path.exists(filepath):
            with open(filepath, 'r') as fp:
                location_data = json.load(fp)

            out_data = []
            for step_key in location_data:
                if step_key.startswith('step_'):
                    image_name = location_data[step_key]['onboard_image'].split('/')[-1].split('.png')[0]
                    if image_name not in latent_data:
                        print('No image found for: ' + step_key)
                    else:
                        line = np.zeros(shape=[common.LATENT_COUNT + 3])
                        line[0] = location_data[step_key]['center_x']
                        line[1] = location_data[step_key]['center_y']
                        line[2] = location_data[step_key]['bearing']
                        line[3:3 + common.LATENT_COUNT] = latent_data[image_name]
                        out_data.append(line)

            if len(out_data) > 0:
                shuffle(out_data)  # very important for test/training split later
                out_data = np.array(out_data)
                print('Current out_data shape: %s' % str(out_data.shape))

                out_file = '../../data/processed/latent_images/position%d.csv' % ifile
                common.mkdir_forfile(out_file)
                np.savetxt(out_file, out_data)
                print('Saved to: ' + out_file)
            else:
                print('Skipped %s' % filepath)
        else:
            break
        ifile += 1

    print('Done.')
