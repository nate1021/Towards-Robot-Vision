import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
import json

import common
import latent_sim.network
import data_collection.camera_meta
import frame_compressor.network

_, _, var_list = latent_sim.network.inference(tf.zeros([1,latent_sim.network.INPUT_DEPTH]), tf.constant(False))

zs, _, _, final_image = frame_compressor.network.encoder(tf.zeros([1, common.IMAGE_HEIGHT, common.IMAGE_WIDTH, common.IMAGE_DEPTH]))
reconstruct = frame_compressor.network.decoder(zs, final_image)
_, restore_frame = frame_compressor.network.get_saver()

trackers_arr = data_collection.camera_meta.load_trackers()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    dic = {}
    with open('../website/saved_model.bin', 'wb') as file:
        dic['latentlength'] = len(trackers_arr)
        dic['trackers'] = trackers_arr
        for pos_i in range(len(trackers_arr)):
            _, restore = latent_sim.network.get_saver(pos_i + 1)
            restore(sess)
            print('Model %d restored.' % (pos_i + 1))

            vals = sess.run(var_list)
            count = 0
            lat_dic = {}
            lat_dic['length'] = len(vals)
            for var in vals:
                file.write(var.astype(np.float32).tostring())
                lat_dic[str(count)] = list(var.shape)
                count += 1
        dic['latent'] = lat_dic

        restore_frame(sess)
        lst = [n for n in tf.all_variables() if n.name.startswith('frame_decoder_') and '/' not in n.name]
        lst.sort(key=lambda x: int(x.name.split(':')[0].split('_')[-1]), reverse=False)

        vals = sess.run(lst)
        dec_dic = {}
        dec_dic['length'] = len(vals)
        count = 0
        for var in vals:
            file.write(var.astype(np.float32).tostring())
            dec_dic[str(count)] = list(var.shape)
            count += 1
        dic['decoder'] = dec_dic

    with open('../website/saved_model.json', 'w') as f:
        json.dump(dic, f)

print('Model exported.')
