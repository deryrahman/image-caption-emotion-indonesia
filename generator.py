from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_random_caption_tokens(ids, tokens):
    result = []

    for i in ids:
        j = np.random.choice(len(tokens[i]))
        result.append(tokens[i][j])

    return result


def batch_generator(batch_size,
                    filenames,
                    transfer_values,
                    tokens,
                    with_transfer_values=True):
    filenames = np.array(filenames)
    while True:
        ids = np.random.randint(len(filenames), size=batch_size)

        if with_transfer_values:
            partial_filenames = filenames[ids]
            partial_transfer_values = []
            for filename in partial_filenames:
                partial_transfer_values.append(transfer_values[filename])
            partial_transfer_values = np.array(partial_transfer_values)

        tokens = get_random_caption_tokens(ids, tokens)

        max_tokens = np.max([len(t) for t in tokens])
        tokens_padded = pad_sequences(
            tokens, maxlen=max_tokens, padding='post', truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        if with_transfer_values:
            x_data = {
                'decoder_input': decoder_input_data,
                'transfer_values_input': partial_transfer_values
            }
        else:
            x_data = {'decoder_input': decoder_input_data}

        y_data = {'decoder_output': decoder_output_data}

        yield (x_data, y_data)
