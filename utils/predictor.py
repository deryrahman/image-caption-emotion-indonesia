from preprocess.images import load_image
from preprocess.tokenizer import mark_end, mark_start
from keras import backend as K


def generate_caption(image_path, tokenizer, rich_model, k=3, max_tokens=30):

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]

    img_size = K.int_shape(rich_model.model_encoder.input)[1:3]

    image = load_image(image_path, size=img_size)

    output_tokens, transfer_values = rich_model.predict(
        image=image,
        token_start=token_start,
        token_end=token_end,
        k=k,
        max_tokens=max_tokens)

    output_text = ''
    for token in output_tokens:
        output_text += tokenizer.token_to_word(token) + ' '
    output_text = output_text.replace(mark_start, '')
    output_text = output_text.replace(mark_end, '')

    return output_tokens, transfer_values, output_text


def seq2seq_generate_caption(transfer_values,
                             input_tokens,
                             tokenizer,
                             rich_model,
                             k=3,
                             max_tokens=30):
    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]

    output_tokens = rich_model.predict(
        transfer_values=transfer_values,
        input_tokens=input_tokens,
        token_start=token_start,
        token_end=token_end,
        k=k,
        max_tokens=max_tokens)

    output_text = ''
    for token in output_tokens:
        output_text += tokenizer.token_to_word(token) + ' '
    output_text = output_text.replace(mark_start, '')
    output_text = output_text.replace(mark_end, '')

    return output_tokens, transfer_values, output_text
