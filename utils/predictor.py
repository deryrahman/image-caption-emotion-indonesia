from preprocess.images import load_image
from preprocess.tokenizer import mark_end, mark_start
from keras import backend as K


def generate_caption(image_path,
                     tokenizer,
                     rich_model,
                     k=3,
                     mode='factual',
                     seq2seq=None,
                     max_tokens=30):

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]

    img_size = K.int_shape(rich_model.model_encoder.input)[1:3]

    image = load_image(image_path, size=img_size)

    output_tokens, transfer_values = rich_model.predict(
        image=image,
        token_start=token_start,
        token_end=token_end,
        max_tokens=max_tokens,
        k=k)

    if seq2seq is not None:
        output_tokens = seq2seq.predict(
            transfer_values=transfer_values,
            input_tokens=output_tokens,
            token_start=token_start,
            token_end=token_end,
            max_tokens=max_tokens,
            k=k)

    output_text = ''
    for token in output_tokens:
        output_text += tokenizer.token_to_word(token) + ' '
    output_text = output_text.replace(mark_start, '')
    output_text = output_text.replace(mark_end, '')

    return image, output_text
