import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import namedtuple
import os
import datetime

Batch = namedtuple('Batch', ['data'])
path = 'http://data.mxnet.io/models/imagenet/resnet/50-layers/'
[mx.test_utils.download(path+'resnet-50-symbol.json'),
 mx.test_utils.download(path+'resnet-50-0000.params'),
 mx.test_utils.download('http://data.mxnet.io/models/imagenet/resnet/synset.txt')]
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)
mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)


with open('../../data/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
csv = open('csv.txt', 'w')
framework = 'mx_net'


def get_image(url, page=False, show=False):
    # download and show the image
    if page:
       fname = mx.test_utils.download(url)
       img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def work(folder):
    for x in os.walk(folder):
        try:
            directory = x[0]
            last_directory_name = os.path.basename(os.path.normpath(directory))
            files = x[2]
            for f in files:
                predict(os.path.join(directory, f), last_directory_name)
        except:
            continue


def predict(img_path, ground_truth):
    img = get_image(img_path, show=True)
    # compute the predict probabilities
    start = datetime.datetime.now()
    mod.forward(Batch([mx.nd.array(img)]))
    end = datetime.datetime.now()
    difference = end - start

    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        s = ','.join(map(str, [labels[i].split(' ')[0], prob[i], img_path, ground_truth, 'mx_net', difference, 'resnet50']))
        print(s)
        csv.write(s)
        csv.write('\n')


def predict_one(url):
    img = get_image(url, show=False)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

work('../../result')
csv.close()