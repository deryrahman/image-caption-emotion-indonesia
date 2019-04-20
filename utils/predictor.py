from preprocess.images import load_image
from preprocess.tokenizer import mark_end, mark_start
from keras.utils.np_utils import np


def generate_caption(image_path,
                     image_model_transfer,
                     decoder_model,
                     tokenizer,
                     img_size,
                     max_tokens=30):

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]

    image = load_image(image_path, size=img_size)
    image_batch = np.expand_dims(image, axis=0)
    transfer_values = image_model_transfer.predict(image_batch)

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    token_int = token_start
    output_text = ''
    count_tokens = 0

    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }
        decoder_output = decoder_model.predict(x_data)

        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer.token_to_word(token_int)

        output_text += " " + sampled_word

        count_tokens += 1

    output_tokens = decoder_input_data[0]
    output_text = output_text.replace(mark_end, '')
    output_text = output_text.strip()

    return image, output_tokens, output_text
