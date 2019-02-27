from keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Input, Dense, LSTM, Embedding, concatenate
from keras.models import Model

IMAGE_SIZE = 224
NUM_CLASSES = 1001


class NIC():

    def __init__(self,
                 token_len,
                 vocab_size,
                 num_image_features=2048,
                 hidden_size=512,
                 embedding_size=512):
        self.token_len = token_len
        self.vocab_size = vocab_size
        self.num_image_features = num_image_features
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.model = None

    def build(self):
        # image embedding
        image_input = Input(
            shape=(1, self.num_image_features), name='image_input')
        image_embedding = Dense(
            self.embedding_size, name='image_embedding')(image_input)

        # word embedding
        word_input = Input(shape=(self.token_len,), name='word_input')
        word_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            name='word_embedding')(word_input)

        # concatenate
        embedding = concatenate([image_embedding, word_embedding],
                                axis=1,
                                name='embedding')

        # decoder LSTM
        decoder_lstm = LSTM(self.hidden_size, name='decoder_lstm')(embedding)
        output = Dense(
            self.vocab_size, activation='softmax', name='output')(decoder_lstm)

        self.model = Model(inputs=[image_input, word_input], outputs=output)

    def get_model(self):
        if not self.model:
            self.build()
        return self.model


class EncoderResNet152():

    def __init__(self, weights='imagenet'):
        self.weights = weights
        self.model = None

    def build(self):
        # from pretrained model
        resnet_152 = ResNet152V2(weights=self.weights)
        self.model = Model(
            inputs=resnet_152.input, outputs=resnet_152.layers[-2].output)

    def get_model(self):
        if not self.model:
            self.build()
        return self.model


class DecoderFactoredLSTM():

    def __init__(self):
        pass

    def build(self):
        pass
