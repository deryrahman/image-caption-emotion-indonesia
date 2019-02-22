import urllib.request
import tensorflow as tf
from model import EncoderResNet152
from data_loader import ImageDecoder
from preprocess import ImagePreprocessor
from tfmodels.research.slim.datasets import imagenet


def get_label(probabilities):
    sorted_inds = []
    names = imagenet.create_readable_names_for_imagenet_labels()
    for probability in probabilities:
        sorted_inds.append(
            [i[0] for i in sorted(enumerate(-probability), key=lambda x: x[1])])

    for i, sorted_ind in enumerate(sorted_inds):
        for j in range(5):
            index = sorted_ind[j]
            print('Probability %0.2f%% => [%s]' %
                  (100 * probabilities[i][index], names[index]))


with tf.Session() as sess:
    # load sample
    url = "https://static.independent.co.uk/s3fs-public/thumbnails/image/2017/03/23/17/electricplane.jpg?w968h681"
    image_bytes = urllib.request.urlopen(url).read()
    images_bytes = tf.constant([image_bytes], dtype=tf.string)

    # decode sample
    image_decoder = ImageDecoder()
    images = image_decoder.decode(images_bytes)

    # preprocess
    image_preprocessor = ImagePreprocessor()
    preprocessed_images = image_preprocessor.preprocess(images)

    # build model
    resnet_152 = EncoderResNet152()
    net, end_point = resnet_152.build(preprocessed_images)
    logits = tf.squeeze(net, [1, 2])
    probabilities = tf.nn.softmax(logits)
    resnet_152.init_fn(sess)
    print('finish build model')

    # predict
    probabilities = sess.run(probabilities)

    get_label(probabilities)
