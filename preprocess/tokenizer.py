from keras.preprocessing.text import Tokenizer

mark_start = 'ssss '
mark_end = ' eeee'


def mark_captions(captions_listlist):
    captions_marked = [[
        mark_start + caption + mark_end for caption in captions_list
    ] for captions_list in captions_listlist]

    return captions_marked


def flatten(captions_listlist):
    captions_list = [
        caption for captions_list in captions_listlist
        for caption in captions_list
    ]

    return captions_list


class TokenizerWrap(Tokenizer):

    def __init__(self, texts, num_words=None):

        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(
            zip(self.word_index.values(), self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]

        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_listlist):
        tokens = [
            self.texts_to_sequences(captions_list)
            for captions_list in captions_listlist
        ]

        return tokens
