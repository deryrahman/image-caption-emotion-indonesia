from model import EncoderResNet152
from keras import backend as K
from images import process_images_all
from tokenizer import mark_captions, flatten, TokenizerWrap, mark_start, mark_end
from preparation import load_caption
from generator import batch_generator
from model import StyleNet
from callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np


def train(dataset, mode, num_words, batch_size, start_epoch, epoch, lstm_layers,
          lstm_units, factored_size, embedding_size, learning_rate, beta_1,
          beta_2, epsilon):

    if lstm_layers == 1:
        fraction = 0.5
    elif lstm_layers == 2:
        fraction = 0.7
    else:
        fraction = 0.8
    layer_size = lstm_layers
    state_size = lstm_units

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    K.tensorflow_backend.set_session(tf.Session(config=config))

    emotions = ['happy', 'sad', 'angry']

    path = './dataset'
    if dataset == 'flickr':
        folder = path + '/flickr10k'
    elif dataset == 'coco':
        folder = path + '/mscoco'
    else:
        raise ValueError(
            'flickr or coco only on dataset. get {}'.format(dataset))

    if mode not in ['factual'] + emotions:
        raise ValueError('mode not support. expected {} get {}'.format(
            ['factual'] + emotions, mode))

    train, val = load_caption(folder, lang='edited')
    filenames_train, captions_train = train
    filenames_train = filenames_train[mode]
    filenames_val, captions_val = val
    filenames_val = filenames_val[mode]
    num_images_train = len(filenames_train)
    print(mode, 'num_images_train', num_images_train)

    encoder_resnet152 = EncoderResNet152()
    img_size = K.int_shape(encoder_resnet152.model.input)[1:3]
    transfer_values_size = K.int_shape(encoder_resnet152.model.output)[1]
    print('img_size', img_size)
    print('transfer_values_size', transfer_values_size)

    transfer_values_train = process_images_all(
        folder_path=folder,
        is_train=True,
        filenames=filenames_train,
        img_size=img_size,
        transfer_values_size=transfer_values_size,
        image_model_transfer=encoder_resnet152.model,
        batch_size=64)
    print("dtype:", transfer_values_train.dtype)
    print("shape:", transfer_values_train.shape)

    transfer_values_val = process_images_all(
        folder_path=folder,
        is_train=False,
        filenames=filenames_val,
        img_size=img_size,
        transfer_values_size=transfer_values_size,
        image_model_transfer=encoder_resnet152.model,
        batch_size=64)
    print("dtype:", transfer_values_val.dtype)
    print("shape:", transfer_values_val.shape)

    captions_train_flat_all = []
    for m in ['factual'] + emotions:
        captions_train_marked = mark_captions(captions_train[m])
        captions_train_flat = flatten(captions_train_marked)
        tokenizer = TokenizerWrap(
            texts=captions_train_flat, num_words=num_words)
        # remove oov words
        tmp = tokenizer.texts_to_sequences(captions_train_flat)
        captions_train_flat = tokenizer.sequences_to_texts(tmp)
        captions_train_flat_all.extend(captions_train_flat)
    tokenizer = TokenizerWrap(texts=captions_train_flat_all)
    num_words = len(tokenizer.word_index)
    print(num_words)

    captions_train_marked = mark_captions(captions_train[mode])
    captions_train_flat = flatten(captions_train_marked)
    tokens_train = tokenizer.captions_to_tokens(captions_train_marked)

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]
    print('token_start', token_start)
    print('token_end', token_end)

    generator = batch_generator(
        batch_size=batch_size,
        num_images_train=num_images_train,
        transfer_values_train=transfer_values_train,
        tokens_train=tokens_train)

    num_captions_train = [len(captions) for captions in captions_train[mode]]
    total_num_captions_train = np.sum(num_captions_train)
    steps_per_epoch = int(total_num_captions_train / batch_size)
    print('steps_per_epoch', steps_per_epoch)

    stylenet = StyleNet(
        mode=mode,
        num_words=num_words,
        transfer_values_size=transfer_values_size,
        state_size=state_size,
        embedding_size=embedding_size,
        factored_size=factored_size,
        lstm_layers=layer_size,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)

    path_checkpoint = 'checkpoints/{}.epoch{}.checkpoint.id.layer{}.factored{}.state{}.embedding{}.keras'.format(
        dataset, epoch, layer_size, factored_size, state_size, embedding_size)
    callback_checkpoint = ModelCheckpoint(stylenet, filepath=path_checkpoint)
    log_dir = (
        './logs/{mode}/'
        '{dataset}_epoch_{start_from}_{to}_layer{layer_size}_factored{factored_size}_'
        'state{state_size}_embedding{embedding_size}').format(
            mode=mode,
            dataset=dataset,
            start_from=start_epoch,
            to=start_epoch + epoch,
            layer_size=layer_size,
            factored_size=factored_size,
            state_size=state_size,
            embedding_size=embedding_size)
    callback_tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=False)

    callbacks = [callback_checkpoint, callback_tensorboard]

    try:
        stylenet.load(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    try:
        stylenet.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epoch,
            callbacks=callbacks)
    except Exception as e:
        print(e)
        print(dataset, mode, num_words, batch_size, epoch, lstm_layers,
              lstm_units, factored_size, embedding_size, learning_rate, beta_1,
              beta_2, epsilon)
        pass


if __name__ == "__main__":
    batch_size = {'factual': [64], 'emotion': [96]}
    epoch = {'factual': [20, 100], 'emotion': [25, 100]}
    lstm_layers = [1, 2, 3]
    lstm_units = [512]
    factored_size = [256, 512, 1024]
    embedding_size = [300]
    learning_rate = {'factual': [0.0002], 'emotion': [0.0005]}
    beta_1 = [0.9]
    beta_2 = [0.999]
    epsilon = [1e-08]

    emotions = ['happy', 'sad', 'angry']
    train(
        dataset='flickr',
        mode='factual',
        num_words=10000,
        batch_size=64,
        start_epoch=0,
        epoch=20,
        lstm_layers=2,
        lstm_units=512,
        factored_size=512,
        embedding_size=300,
        learning_rate=0.0002,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)
