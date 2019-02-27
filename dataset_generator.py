import numpy as np
import keras


def data_process(batch_size, max_length, vocab_len, tokenizer, img_features,
                 mapping_desc):
    count = 0
    inputs_img = []
    inputs_words = []
    outputs = []
    while True:
        for k, v in mapping_desc.items():
            curr_img = img_features[k]
            curr_caps = mapping_desc[k]
            curr_words_list = []
            next_words_list = []

            # [img, <start>, x[0], ... , x[-1]]
            # [<start>, x[0], x[1], ... , <end>]
            for i, caption in enumerate(curr_caps):
                curr_words_list.append(caption[:-1])  # remove <end> token
                next_words_list.append(caption)

            curr_words_list = tokenizer.texts_to_sequences(curr_words_list)
            next_words_list = tokenizer.texts_to_sequences(next_words_list)

            curr_words_list = keras.preprocessing.sequence.pad_sequences(
                curr_words_list, maxlen=max_length, padding='post')
            next_words_list = keras.preprocessing.sequence.pad_sequences(
                next_words_list, maxlen=max_length + 1, padding='post')

            for curr_words, next_words in zip(curr_words_list, next_words_list):
                labels = keras.utils.to_categorical(
                    next_words, num_classes=vocab_len)
                inputs_img.append(curr_img)
                inputs_words.append(curr_words)
                outputs.append(labels)
            if count < batch_size:
                count += 1
                continue
            yield [[np.array(inputs_img),
                    np.array(inputs_words)],
                   np.array(outputs)]
            count = 0
            inputs_img = []
            inputs_words = []
            outputs = []
