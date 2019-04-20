from preparation import load_caption
from model import EncoderResNet152
from keras import backend as K
from images import process_images_all
from tokenizer import mark_captions, flatten, TokenizerWrap
from generator import batch_generator
from model import StyleNet
from callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
from evaluator import bleu_evaluator
from predictor import generate_caption
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
import numpy as np
import argparse
import os


def main(args):
    epoch_num = args.epoch_num
    injection_mode = args.injection_mode
    checkpoints_path = args.checkpoints_path
    dataset_path = args.dataset_path
    dataset = args.dataset
    logs_path = args.logs_path
    mode = args.mode
    load_model = args.load_model
    gpu_frac = args.gpu_frac
    with_transfer_value = args.with_transfer_value

    state_size = args.state_size
    embedding_size = args.embedding_size
    factored_size = args.factored_size
    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    epsilon = args.epsilon
    lstm_layers = args.lstm_layers
    batch_size = args.batch_size
    num_words = args.num_words

    config = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    K.tensorflow_backend.set_session(tf.Session(config=config))

    modes = ['happy', 'sad', 'angry']
    captions = []
    filenames = []
    for md in modes + ['factual']:
        train, val, test = load_caption(dataset_path + '/' + md)
        filenames_train, captions_train = train
        filenames_val, captions_val = val
        filenames_test, captions_test = test
        if md == 'factual':
            filenames += filenames_train + filenames_val + filenames_test
        captions += captions_train + captions_val + captions_test

    encoder_resnet152 = EncoderResNet152()

    img_size = K.int_shape(encoder_resnet152.model.input)[1:3]
    transfer_values_size = K.int_shape(encoder_resnet152.model.output)[1]
    print('img_size', img_size)
    print('transfer_values_size', transfer_values_size)

    transfer_values = process_images_all(
        folder_path=dataset_path,
        filenames=filenames,
        img_size=img_size,
        transfer_values_size=transfer_values_size,
        image_model_transfer=encoder_resnet152.model,
        batch_size=64)
    print('transfer_values count', len(transfer_values))

    captions_marked = mark_captions(captions)
    captions_flat = flatten(captions_marked)
    tokenizer = TokenizerWrap(texts=captions_flat, num_words=num_words)
    print('num_words', num_words)

    train, val, test = load_caption(dataset_path + '/' + mode)
    filenames_train, captions_train = train
    filenames_val, captions_val = val
    filenames_test, captions_test = test

    # for local testing only
    filenames_train, captions_train = filenames_train[:50], captions_train[:50]
    filenames_val, captions_val = filenames_val[:5], captions_val[:5]
    filenames_test, captions_test = filenames_test[:5], captions_test[:5]

    num_captions_train = [len(captions) for captions in captions_train]
    total_num_captions_train = np.sum(num_captions_train)
    num_captions_val = [len(captions) for captions in captions_val]
    total_num_captions_val = np.sum(num_captions_val)
    num_captions_test = [len(captions) for captions in captions_test]
    total_num_captions_test = np.sum(num_captions_test)
    train_steps = int(total_num_captions_train / batch_size)
    val_steps = int(total_num_captions_val / batch_size)
    test_steps = int(total_num_captions_test / batch_size)
    print('train steps', train_steps)
    print('val steps', val_steps)
    print('test steps', test_steps)

    captions_train_marked = mark_captions(captions_train)
    tokens_train = tokenizer.captions_to_tokens(captions_train_marked)

    captions_val_marked = mark_captions(captions_val)
    tokens_val = tokenizer.captions_to_tokens(captions_val_marked)

    captions_test_marked = mark_captions(captions_test)
    tokens_test = tokenizer.captions_to_tokens(captions_test_marked)

    if with_transfer_value == 1:
        transfer_values_train = np.array(
            [transfer_values[filename] for filename in filenames_train])
        transfer_values_val = np.array(
            [transfer_values[filename] for filename in filenames_val])
        transfer_values_test = np.array(
            [transfer_values[filename] for filename in filenames_test])

    generator_train = batch_generator(
        batch_size=batch_size,
        transfer_values=transfer_values_train
        if with_transfer_value == 1 else None,
        tokens=tokens_train,
        with_transfer_values=with_transfer_value == 1)

    generator_val = batch_generator(
        batch_size=batch_size,
        transfer_values=transfer_values_val
        if with_transfer_value == 1 else None,
        tokens=tokens_val,
        with_transfer_values=with_transfer_value == 1)

    generator_test = batch_generator(
        batch_size=batch_size,
        transfer_values=transfer_values_test
        if with_transfer_value == 1 else None,
        tokens=tokens_test,
        with_transfer_values=with_transfer_value == 1)

    stylenet = StyleNet(
        injection_mode=injection_mode,
        num_words=num_words,
        trainable_factor=mode == 'factual',
        include_transfer_value=with_transfer_value == 1,
        mode=mode,
        state_size=state_size,
        embedding_size=embedding_size,
        factored_size=factored_size,
        lstm_layers=lstm_layers,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)

    if not os.path.exists(checkpoints_path + '/stylenet'):
        os.mkdir(checkpoints_path + '/stylenet')

    path_checkpoint = checkpoints_path + (
        '/stylenet/{}.checkpoint.id.injection{}.layer{}.factored{}.state{}.embedding{}.keras'
    ).format(dataset, injection_mode, lstm_layers, factored_size, state_size,
             embedding_size)
    callback_checkpoint = ModelCheckpoint(
        stylenet,
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)
    callback_earystoping = EarlyStopping(
        monitor='val_loss', verbose=1, patience=10)
    log_dir = (
        logs_path + '/stylenet/{mode}/{injection}/'
        '{dataset}_epoch_{start_from}_{to}_layer{layer_size}_factored{factored_size}_'
        'state{state_size}_embedding{embedding_size}').format(
            mode=mode,
            injection=injection_mode,
            dataset=dataset,
            start_from=0,
            to=epoch_num,
            layer_size=lstm_layers,
            factored_size=factored_size,
            state_size=state_size,
            embedding_size=embedding_size)
    callback_tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=False)

    callbacks = [
        callback_checkpoint, callback_earystoping, callback_tensorboard
    ]

    if load_model == 1:
        try:
            stylenet.load(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

    stylenet.model.fit_generator(
        generator=generator_train,
        steps_per_epoch=train_steps,
        epochs=epoch_num,
        validation_data=generator_val,
        validation_steps=val_steps,
        callbacks=callbacks)

    scores = stylenet.model.evaluate_generator(
        generator=generator_test, steps=test_steps, verbose=1)
    print('test loss', scores)

    stylenet = StyleNet(
        injection_mode=injection_mode,
        num_words=num_words + 1,
        include_transfer_value=True,
        mode=mode,
        trainable_factor=mode == 'factual',
        state_size=state_size,
        embedding_size=embedding_size,
        factored_size=factored_size,
        lstm_layers=lstm_layers,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)
    try:
        stylenet.load(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    references = []
    predictions = []
    for filename, refs in zip(filenames_test, captions_test):
        _, _, output_text = generate_caption(
            image_path=dataset_path + '/img/' + filename,
            image_model_transfer=encoder_resnet152.model,
            decoder_model=stylenet.model,
            tokenizer=tokenizer,
            img_size=img_size)
        print(output_text)
        predictions.append(output_text)
        references.append(refs)
    print(bleu_evaluator(references, predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='StyleNet Indonesia: Generate Image Commenting With Emotions'
    )
    parser.add_argument(
        '--load_model',
        type=int,
        default=0,
        help='load saved model from checkpoints or not (1, 0)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='flickr',
        help='dataset either flickr or coco')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset/flickr10k',
        help='path for dataset')
    parser.add_argument(
        '--checkpoints_path',
        type=str,
        default='./checkpoints',
        help='path for save checkpoints model')
    parser.add_argument(
        '--logs_path',
        type=str,
        default='./logs',
        help='path for Tensorboard logging')
    parser.add_argument(
        '--num_words',
        type=int,
        default=10000,
        help='num words used for embedding layer')
    parser.add_argument(
        '--mode',
        type=str,
        default='factual',
        help='emotion mode; factual, happy, sad, angry')
    parser.add_argument(
        '--injection_mode',
        type=str,
        default='init',
        help='transfer value injection mode')
    parser.add_argument(
        '--with_transfer_value',
        type=int,
        default=1,
        help='use transfer value or not')
    parser.add_argument(
        '--gpu_frac',
        type=float,
        default=0.5,
        help='gpu fraction. How much GPU resources will be used')
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=20,
        help='number of epoch used for training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='number of batch size used for training')
    parser.add_argument(
        '--lstm_layers',
        type=int,
        default=1,
        help='number of LSTM layer in the network')
    parser.add_argument(
        '--state_size',
        type=int,
        default=512,
        help='number of LSTM state units')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=300,
        help='number of embedding size used for LSTM')
    parser.add_argument(
        '--factored_size',
        type=int,
        default=512,
        help='number of Factored LSTM state units')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002,
        help='used for learning rate')
    parser.add_argument(
        '--beta_1',
        type=float,
        default=0.9,
        help='used for beta 1 parameter of LSTM training')
    parser.add_argument(
        '--beta_2',
        type=float,
        default=0.999,
        help='used for beta 1 parameter of LSTM training')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-08,
        help='used for epsilon parameter of LSTM training')
    args = parser.parse_args()
    for k in args.__dict__:
        print(k, '=', args.__dict__[k])
    main(args)
