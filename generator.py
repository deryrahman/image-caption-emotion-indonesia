from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_random_caption_tokens(idx, tokens_train):
    result = []

    for i in idx:
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)

    return result


def batch_generator(batch_size, num_images_train, transfer_values_train,
                    tokens_train):
    while True:
        idx = np.random.randint(num_images_train, size=batch_size)
        transfer_values = transfer_values_train[idx]

        tokens = get_random_caption_tokens(idx, tokens_train)

        max_tokens = np.max([len(t) for t in tokens])
        tokens_padded = pad_sequences(
            tokens, maxlen=max_tokens, padding='post', truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        y_data = {'decoder_output': decoder_output_data}

        yield (x_data, y_data)
