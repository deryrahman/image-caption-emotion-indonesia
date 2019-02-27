import numpy as np
import keras


def data_process(batch_size, max_length, vocab_len, tokenizer, img_features,
                 mapping_desc):
    count = 0
    batch_img = []
    batch_text = []
    batch_label = []
    while True:
        for k, texts in mapping_desc.items():
            curr_img = img_features[k]
            for text in texts:
                text = text.split()
                for i in range(len(text) - 1):
                    count += 1
                    curr_words = ' '.join(text[:i + 1])
                    label = text[i + 1]

                    batch_img.append(curr_img)
                    batch_text.append(curr_words)
                    batch_label.append(label)

                    if count >= batch_size:

                        batch_text = tokenizer.texts_to_sequences(batch_text)
                        batch_text = keras.preprocessing.sequence.pad_sequences(
                            batch_text, maxlen=max_length)
                        batch_label = tokenizer.texts_to_sequences(batch_label)
                        batch_label = keras.utils.to_categorical(
                            batch_label, num_classes=vocab_len)

                        yield [[np.array(batch_img),
                                np.array(batch_text)],
                               np.array(batch_label)]

                        count = 0
                        batch_img = []
                        batch_text = []
                        batch_label = []
