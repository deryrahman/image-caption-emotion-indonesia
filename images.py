from PIL import Image
import numpy as np
import sys
import os
import pickle


def load_image(path, size=None):
    img = Image.open(path)

    if size is not None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    img = np.array(img)
    img = img / 255.0

    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


def print_progress(count, max_count):
    msg = "\r- Progress: {0:.1%}".format(count / max_count)
    sys.stdout.write(msg)
    sys.stdout.flush()


def process_images(folder_path,
                   filenames,
                   img_size,
                   transfer_values_size,
                   image_model_transfer,
                   batch_size=32):

    num_images = len(filenames)

    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    shape = (num_images, transfer_values_size)
    values = np.zeros(shape=shape, dtype=np.float16)

    start_index = 0

    while start_index < num_images:

        print_progress(count=start_index, max_count=num_images)

        end_index = start_index + batch_size
        end_index = num_images if end_index > num_images else end_index

        current_batch_size = end_index - start_index

        for i, filename in enumerate(filenames[start_index:end_index]):
            path = folder_path + '/img/' + filename
            img = load_image(path, size=img_size)
            image_batch[i] = img

        values[start_index:end_index] = image_model_transfer.predict(
            image_batch[:current_batch_size])

        start_index = end_index
    print()

    return values


def process_images_all(folder_path,
                       filenames,
                       img_size,
                       transfer_values_size,
                       image_model_transfer,
                       batch_size=32):
    print("Processing {} images ...".format(len(filenames)))

    path = folder_path + '/cache'
    cache_path = path + ("/transfer_values.pkl")

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            transfer_values = pickle.load(f)
    else:
        values = process_images(folder_path, filenames, img_size,
                                transfer_values_size, image_model_transfer,
                                batch_size)
        transfer_values = {}
        for filename, val in zip(filenames, values):
            transfer_values[filename] = val
        with open(cache_path, 'wb') as f:
            pickle.dump(transfer_values, f)

    return transfer_values
