import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from random import randint

import common
import frame_compressor.train

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    frame_compressor.train.restore(sess)
    print('Model restored.')

    frame_compressor.train.run_tests(sess)

    sess.run(frame_compressor.train.dataset_init_op, feed_dict={frame_compressor.train.validation_placeholder: True})
    abs_max = 3

    root = Tk()
    root.title('VAE')
    root.geometry('{}x{}'.format(1000, 700))

    TOP_HEIGHT = 560
    TOP_TRACK_LENGTH = 300
    # create all of the main containers
    top_frame = Frame(root, bg='cyan', width=1, height=TOP_HEIGHT)
    center = Frame(root, width=1, height=50)

    # layout all of the main containers
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    top_frame.grid(row=0, sticky="nsew")
    center.grid(row=1, sticky="ew")

    # create the center widgets
    top_frame.grid_rowconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)

    top_left = Frame(top_frame, bg='grey2', width=TOP_TRACK_LENGTH + 50, height=TOP_HEIGHT)
    top_right = Frame(top_frame, bg='black', width=2000, height=TOP_HEIGHT)

    top_left.grid(row=0, column=0, sticky="ns")
    top_right.grid(row=0, column=1, sticky="nsew")

    # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel1 = Label(top_right, image=None)
    panel1.grid(row=0, column=0, sticky="ew")


    def slide_changed(tkinter_Event):
        z_vec = np.zeros(shape=[1, common.LATENT_COUNT])
        for i in range(common.LATENT_COUNT):
            z_vec[0,i] = (sliders[i].get() / 100.0) * abs_max
        recs = sess.run(frame_compressor.train.output_reconstruct, feed_dict={frame_compressor.train.z: z_vec})
        update_image(recs[0])


    sliders = []
    for i in range(common.LATENT_COUNT):
        w1 = Scale(top_left, from_=-100, to=100, length=TOP_TRACK_LENGTH, orient=HORIZONTAL)
        w1.set(0)
        w1.grid(row=i % 12, column=int(i / 12))
        w1.bind("<ButtonRelease-1>", slide_changed)
        sliders.append(w1)

    def update_image(image_matrix):
        img = Image.fromarray(image_matrix.astype(np.uint8))
        img = img.resize((400, 300))
        img = ImageTk.PhotoImage(img)
        panel1.configure(image=img)
        panel1.image = img

    def load_random_image():
        zs, recs = sess.run((frame_compressor.train.z, frame_compressor.train.output_reconstruct))
        for i in range(common.LATENT_COUNT):
            sliders[i].set((zs[0][i] / abs_max) * 100)
        update_image(recs[0])

    def random_slides():
        for i in range(common.LATENT_COUNT):
            sliders[i].set(20 * np.random.normal(loc=0.0, scale=1.0))
        slide_changed(None)

    def random_slide():
        index = randint(0, common.LATENT_COUNT - 1)
        sliders[index].set(randint(-100, 100))
        slide_changed(None)


    Button(center, text='Load Random Image', command=load_random_image).grid(row=0, column=0, padx=5)
    Button(center, text='Randomise Latent Variables', command=random_slides).grid(row=0, column=1, padx=5)
    Button(center, text='Randomise Single Latent Variable', command=random_slide).grid(row=0, column=2, padx=5)

    root.mainloop()
