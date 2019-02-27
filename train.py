from dataset_generator import data_process
from model import NIC
from preparation import load
import keras

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
#keras.backend.set_session(sess)

checkpoint_path = './model-ep005-loss0.792-val_loss0.787.h5'
flickr_path = './dataset/flickr10k/flickr_10k.json'
image_path = './dataset/flickr10k/img/'
features_path = './pretrained/features.pkl'
caption_data_path = './dataset/flickr10k/caption.pkl'
train_data_path = './dataset/flickr10k/train.pkl'
validation_data_path = './dataset/flickr10k/validation.pkl'

batch_size = 50
num_epoch = 50

features = load(features_path)
caption_data_en = load(caption_data_path + '.en')
train_data_en = load(train_data_path + '.en')
validation_data_en = load(validation_data_path + '.en')


def to_vocab(mapping):
    vocab = set()
    for k, v_list in mapping.items():
        for v in v_list:
            for val in v.split():
                vocab.add(val)
    return list(vocab)


def create_tokenizer(desc):
    lines = [l for line in desc.values() for l in line]
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


vocab_en = to_vocab(caption_data_en)
tokenizer_en = create_tokenizer(caption_data_en)

max_length_en = max([
    len(cap.split())
    for captions in caption_data_en.values()
    for cap in captions
])

nic = NIC(token_len=max_length_en, vocab_size=len(vocab_en))
model = nic.get_model()
model.load_weights(checkpoint_path)
model.compile(
    loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

train_generator = data_process(batch_size, max_length_en, len(vocab_en),
                               tokenizer_en, features, train_data_en)
validation_generator = data_process(batch_size, max_length_en, len(vocab_en),
                                    tokenizer_en, features, validation_data_en)
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_data_en) // batch_size,
    epochs=num_epoch,
    callbacks=[checkpoint],
    validation_data=validation_generator,
    validation_steps=len(validation_data_en) // batch_size)
