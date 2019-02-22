import tensorflow as tf
import glob
tf.train.string_input_producer


class ImageDecoder():
    """Helper class for decoding images"""

    def decode(self, images_bytes):
        """Decode tensor images_bytes dtype=tf.string with rank 2, dimension (None, None)
        into tensor images with rank 4, dimension (None, None, None, 3), [batch, height, width, channels]

        Arguments:
            images_bytes {tensor} -- tensor string dtype=tf.string with dimension [batch, bytes]
        
        Returns:
            images {tensor} -- tensor rank 4 [batch, height, width, channels]
        """

        func = lambda x: tf.image.decode_jpeg(x, channels=3)
        images = tf.map_fn(func, images_bytes, dtype=tf.uint8)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        return images


class ImageLoader():
    """Helper class for load image"""

    def load(self, path_folder, extension='jpg'):
        """Load image from path_folder. Currently only support jpg
        
        Arguments:
            path_folder {str} -- path folder
            extension {str} -- extension, eg. jpg (default: {'jpg'})
        
        Returns:
            image_bytes {tensor} -- tensor rank 2 [batch, bytes]
        """

        if extension != 'jpg':
            raise Exception('{} must be jpg'.format(extension))
        filenames = glob.glob('{}/*.{}'.format(path_folder, extension))
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        images_bytes = tf.map_fn(tf.read_file, filenames)
        return images_bytes


# image_loader = ImageLoader()
# a = image_loader.load(path_folder='./dataset/example/')
# image_decoder = ImageDecoder()
# b = image_decoder.decode(a)
# print(b)