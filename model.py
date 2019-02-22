import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

IMAGE_SIZE = 224
NUM_CLASSES = 1001


class EncoderResNet152():

    def __init__(self):
        self.checkpoint = "./pretrained/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt"

    def build(self, preprocessed_images):
        assert len(preprocessed_images.shape) == 4
        assert preprocessed_images.shape[1] == IMAGE_SIZE
        assert preprocessed_images.shape[2] == IMAGE_SIZE
        assert preprocessed_images.shape[3] == 3
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, end_point = resnet_v2.resnet_v2_152(
                preprocessed_images, num_classes=NUM_CLASSES, is_training=False)
        self.init_fn = slim.assign_from_checkpoint_fn(
            self.checkpoint, slim.get_variables_to_restore())
        return net, end_point


class FactoredLSTM():

    def __init__(self):
        pass
