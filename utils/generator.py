from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_random_caption_tokens(ids, tokens):
    result = []

    for i in ids:
        j = np.random.choice(len(tokens[i]))
        result.append(tokens[i][j])

    return result


def stylenet_batch_generator(batch_size,
                             transfer_values,
                             tokens,
                             with_transfer_values=True):
    while True:
        if with_transfer_values:
            ids = np.random.randint(len(tokens), size=batch_size)
            partial_transfer_values = transfer_values[ids]

            partial_tokens = get_random_caption_tokens(ids, tokens)

            max_tokens = np.max([len(t) for t in partial_tokens])
            tokens_padded = pad_sequences(
                partial_tokens,
                maxlen=max_tokens,
                padding='post',
                truncating='post')

            decoder_input_data = tokens_padded[:, 0:-1]
            decoder_output_data = tokens_padded[:, 1:]

            x_data = {
                'decoder_input': decoder_input_data,
                'transfer_values_input': partial_transfer_values
            }

            y_data = {'decoder_output': decoder_output_data}

            yield (x_data, y_data)
        else:
            ids = np.random.randint(len(tokens), size=batch_size)
            partial_tokens = get_random_caption_tokens(ids, tokens)
            max_tokens = np.max([len(t) for t in partial_tokens])

            tokens_padded = pad_sequences(
                partial_tokens,
                maxlen=max_tokens,
                padding='post',
                truncating='post')

            decoder_input_data = tokens_padded[:, 0:-1]
            decoder_output_data = tokens_padded[:, 1:]

            x_data = {'decoder_input': decoder_input_data}

            y_data = {'decoder_output': decoder_output_data}

            yield (x_data, y_data)


def seq2seq_batch_generator(batch_size, transfer_values, tokens_encoder_input,
                            tokens_decoder_input):
    while True:
        ids = np.random.randint(len(tokens_encoder_input), size=batch_size)

        partial_transfer_values = transfer_values[ids]
        partial_tokens_encoder_input = get_random_caption_tokens(
            ids, tokens_encoder_input)
        partial_tokens_decoder_input = get_random_caption_tokens(
            ids, tokens_decoder_input)

        max_tokens_encoder = np.max(
            [len(t) for t in partial_tokens_encoder_input])
        max_tokens_decoder = np.max(
            [len(t) for t in partial_tokens_decoder_input])

        tokens_encoder_padded = pad_sequences(
            partial_tokens_encoder_input,
            maxlen=max_tokens_encoder,
            padding='post',
            truncating='post')
        tokens_decoder_padded = pad_sequences(
            partial_tokens_decoder_input,
            maxlen=max_tokens_decoder,
            padding='post',
            truncating='post')

        encoder_input_data = tokens_encoder_padded
        decoder_input_data = tokens_decoder_padded[:, 0:-1]
        decoder_output_data = tokens_decoder_padded[:, 1:]

        x_data = {
            'seq2seq_transfer_values_input': partial_transfer_values,
            'seq2seq_encoder_input': encoder_input_data,
            'seq2seq_decoder_input': decoder_input_data
        }

        y_data = {'seq2seq_decoder_output': decoder_output_data}

        yield (x_data, y_data)
