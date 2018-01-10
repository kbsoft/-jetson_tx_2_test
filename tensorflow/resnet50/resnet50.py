from __future__ import division
from __future__ import print_function

import os
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
import tensorflow.contrib.slim.nets

# https://github.com/tensorflow/models/tree/master/research/slim

csv = open('csv.txt', 'w')
framework = 'tensorflow'
with open('../../data/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


class Network:
    WIDTH = 224
    HEIGHT = 224
    CLASSES = 1000

    def __init__(self, checkpoint, threads):
        # Create the session
        self.session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                          intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():
            # Construct the model
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 3])

            with tf_slim.arg_scope(tf_slim.nets.resnet_v1.resnet_arg_scope(is_training=False)):
                resnet, _ = tf_slim.nets.resnet_v1.resnet_v1_50(self.images, num_classes=self.CLASSES)

            self.predictions = tf.squeeze(resnet, [1, 2])[0]

            # Load the checkpoint
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, checkpoint)

            # JPG loading
            self.jpeg_file = tf.placeholder(tf.string, [])
            self.jpeg_data = tf.image.resize_image_with_crop_or_pad(
                tf.image.decode_jpeg(tf.read_file(self.jpeg_file), channels=3), self.HEIGHT, self.WIDTH)

    def load_jpeg(self, jpeg_file):
        return self.session.run(self.jpeg_data, {self.jpeg_file: jpeg_file})

    def predict(self, image):
        print(self.predictions.shape)
        return self.session.run(self.predictions, {self.images: [image]})


def work(folder, network):
    count = 0
    for x in os.walk(folder):
        try:
            directory = x[0]
            last_directory_name = os.path.basename(os.path.normpath(directory))
            files = x[2]
            count += 1
            for f in files:
                predict(os.path.join(directory, f), network, last_directory_name)
        except:
            continue


def predict(image_file, network, ground_truth):
    image_data = network.load_jpeg(image_file)
    start = datetime.datetime.now()
    prediction = network.predict(image_data)
    end = datetime.datetime.now()
    difference = end - start
    arr = np.array(prediction)
    for i in list(arr.argsort()[-5:][::-1]):
        predicted_class = labels[i].split(' ')[0]
        predicted_procent = prediction[i] / 100
        s = ','.join(map(str, [predicted_class, predicted_procent, image_file, ground_truth, framework, difference, 'resnet50']))
        print(s)
        csv.write(s)
        csv.write('\n')


if __name__ == "__main__":
    network = Network('../../data/resnet_v1_50.ckpt', 1)
    work('../../result', network)
    csv.close()
