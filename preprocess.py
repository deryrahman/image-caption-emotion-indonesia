import tensorflow as tf
from tfmodels.research.slim.preprocessing import inception_preprocessing

IMAGE_SIZE = 224


class ImagePreprocessor():
    """Helper class for preprocess image before feed into ResNet152"""

    def preprocess(self, images):
        func = lambda x: inception_preprocessing.preprocess_image(
            x, IMAGE_SIZE, IMAGE_SIZE)
        preprocessed_image = tf.map_fn(func, images, dtype=tf.float32)
        return tf.stack(preprocessed_image, axis=0)
