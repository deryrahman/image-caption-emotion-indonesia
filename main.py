import urllib.request
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim
from tensorflow.contrib.slim.nets import resnet_v2
from tfmodels.research.slim.preprocessing import inception_preprocessing
from tfmodels.research.slim.datasets import imagenet

IMAGE_SIZE = 224
NUM_CLASSES = 1001
graph = tf.Graph()


with graph.as_default():
    # image for sampling
    image_bytes = tf.placeholder(tf.string)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    preprocessed_image = inception_preprocessing.preprocess_image(
        image, IMAGE_SIZE, IMAGE_SIZE)
    preprocessed_images = tf.expand_dims(preprocessed_image, 0)

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_point = resnet_v2.resnet_v2_152(
            preprocessed_images, num_classes=NUM_CLASSES, is_training=False)
    logits = tf.squeeze(net, [1, 2])
    probabilities = tf.nn.softmax(logits)

    checkpoint = "./model/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt"
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint, slim.get_variables_to_restore())

with tf.Session(graph=graph) as sess:
    # restore from checkpoint
    init_fn(sess)

    # load sample
    url = "https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg"
    image_bytes_1 = urllib.request.urlopen(url).read()

    # predict sample
    probabilities = sess.run(probabilities, feed_dict={
                             image_bytes: image_bytes_1})
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0]
                   for i in sorted(enumerate(-probabilities),
                                   key=lambda x:x[1])]
    names = imagenet.create_readable_names_for_imagenet_labels()
    result_text = ''
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' %
              (100*probabilities[index], names[index]))
    result_text += str(names[sorted_inds[0]])+'=>' + \
        str("{0:.2f}".format(100*probabilities[sorted_inds[0]]))+'%\n'
