from preprocess.dataset import load_caption
from model import StyleNet, Seq2Seq
from preprocess.tokenizer import mark_captions
from utils.generator import stylenet_batch_generator, seq2seq_batch_generator
from utils.evaluator import bleu_evaluator
from utils.predictor import generate_caption
from callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping
# from tensorflow.core.protobuf import rewriter_config_pb2
from keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import os
import pickle


def assert_path_error(path):
    if not os.path.exists(path):
        raise ValueError(path + ' not found')


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
    with_attention = args.with_attention
    emotion_training_mode = args.emotion_training_mode

    state_size = args.state_size
    embedding_size = args.embedding_size
    factored_size = args.factored_size
    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    epsilon = args.epsilon
    lstm_layers = args.lstm_layers
    batch_size = args.batch_size
    early_stop = args.early_stop
    beam_search = args.beam_search
    dropout = args.dropout

    assert_path_error(dataset_path)
    assert_path_error(dataset_path + '/' + mode)
    assert_path_error(dataset_path + '/img')
    assert_path_error(dataset_path + '/captions.json')
    assert_path_error(dataset_path + '/cache/tokenizer.pkl')
    assert_path_error(dataset_path + '/cache/transfer_values.pkl')

    if not os.path.exists(checkpoints_path + '/stylenet'):
        os.mkdir(checkpoints_path + '/stylenet')

    K.clear_session()
    config = tf.ConfigProto()
    # off = rewriter_config_pb2.RewriterConfig.OFF
    # config.graph_options.rewrite_options.memory_optimization = off

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    K.tensorflow_backend.set_session(tf.Session(config=config))

    with open(dataset_path + '/cache/transfer_values.pkl', 'rb') as f:
        transfer_values = pickle.load(f)
    print('transfer_values count', len(transfer_values))

    with open(dataset_path + '/cache/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    num_words = tokenizer.num_words
    print('num_words', num_words)

    path_checkpoint = checkpoints_path + (
        '/stylenet/{dataset}.checkpoint.id.injection{injection_mode}'
        '.layer{lstm_layers}.factored{factored_size}.state{state_size}.embedding{embedding_size}.keras'
    ).format(
        dataset=dataset,
        injection_mode=injection_mode,
        lstm_layers=lstm_layers,
        factored_size=factored_size,
        state_size=state_size,
        embedding_size=embedding_size)
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

    if mode == 'factual':
        train, val, test = load_caption(dataset_path + '/factual')

        filenames_train, captions_train = train
        filenames_val, captions_val = val
        filenames_test, captions_test = test

        # # for local testing only
        # filenames_train = filenames_train[:50]
        # captions_train = captions_train[:50]
        # filenames_val = filenames_val[:5]
        # captions_val = captions_val[:5]
        # filenames_test = filenames_test[:5]
        # captions_test = captions_test[:5]

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

        transfer_values_train = np.array(
            [transfer_values[filename] for filename in filenames_train])
        transfer_values_val = np.array(
            [transfer_values[filename] for filename in filenames_val])
        transfer_values_test = np.array(
            [transfer_values[filename] for filename in filenames_test])

        generator_train = stylenet_batch_generator(
            batch_size=batch_size,
            transfer_values=transfer_values_train,
            tokens=tokens_train,
            with_transfer_values=with_transfer_value == 1)

        generator_val = stylenet_batch_generator(
            batch_size=batch_size,
            transfer_values=transfer_values_val,
            tokens=tokens_val,
            with_transfer_values=with_transfer_value == 1)

        generator_test = stylenet_batch_generator(
            batch_size=batch_size,
            transfer_values=transfer_values_test,
            tokens=tokens_test,
            with_transfer_values=with_transfer_value == 1)

        stylenet = StyleNet(
            injection_mode=injection_mode,
            num_words=num_words,
            trainable_model=True,
            mode=mode,
            state_size=state_size,
            embedding_size=embedding_size,
            factored_size=factored_size,
            lstm_layers=lstm_layers,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            dropout=dropout)

        if load_model == 1:
            try:
                stylenet.load(path_checkpoint)
            except Exception as error:
                print("Error trying to load checkpoint.")
                print(error)

        callback_checkpoint = ModelCheckpoint(
            stylenet,
            filepath=path_checkpoint,
            monitor='val_loss',
            verbose=1,
            save_best_only=True)

        callback_tensorboard = TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=False)

        callbacks = [callback_checkpoint, callback_tensorboard]
        if early_stop > 0:
            callback_earystoping = EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=early_stop,
                restore_best_weights=True)
            callbacks = [
                callback_checkpoint, callback_earystoping, callback_tensorboard
            ]

        stylenet.model_decoder.fit_generator(
            generator=generator_train,
            steps_per_epoch=train_steps,
            epochs=epoch_num,
            validation_data=generator_val,
            validation_steps=val_steps,
            callbacks=callbacks)

        scores = stylenet.model_decoder.evaluate_generator(
            generator=generator_test, steps=test_steps, verbose=1)
        print('test loss', scores)

        references = []
        predictions = []
        for filename, refs in zip(filenames_test, captions_test):
            _, output_text = generate_caption(
                image_path=dataset_path + '/img/' + filename,
                tokenizer=tokenizer,
                stylenet=stylenet,
                k=beam_search,
                mode='factual')
            predictions.append(output_text)
            references.append(refs)
        print(bleu_evaluator(references, predictions))
    else:
        train, val, test = load_caption(dataset_path + '/factual')
        filenames_train, captions_train = train
        filenames_val, captions_val = val
        filenames_test, captions_test = test
        filenames_factual = filenames_train + filenames_val + filenames_test
        captions_factual = captions_train + captions_val + captions_test

        train, val, test = load_caption(dataset_path + '/' + mode)

        filenames_train, captions_train = train
        filenames_val, captions_val = val
        filenames_test, captions_test = test

        # # for local testing only
        # filenames_train = filenames_train[:50]
        # captions_train = captions_train[:50]
        # filenames_val = filenames_val[:5]
        # captions_val = captions_val[:5]
        # filenames_test = filenames_test[:5]
        # captions_test = captions_test[:5]

        mp = {}
        for filename, caption in zip(filenames_factual, captions_factual):
            mp[filename] = caption

        encoder_input_train = [mp[filename] for filename in filenames_train]
        decoder_input_train = captions_train
        encoder_input_val = [mp[filename] for filename in filenames_val]
        decoder_input_val = captions_val
        encoder_input_test = [mp[filename] for filename in filenames_test]
        decoder_input_test = captions_test

        encoder_input_train_marked = mark_captions(encoder_input_train)
        tokens_encoder_input_train = tokenizer.captions_to_tokens(
            encoder_input_train_marked)
        decoder_input_train_marked = mark_captions(decoder_input_train)
        tokens_decoder_input_train = tokenizer.captions_to_tokens(
            decoder_input_train_marked)

        encoder_input_val_marked = mark_captions(encoder_input_val)
        tokens_encoder_input_val = tokenizer.captions_to_tokens(
            encoder_input_val_marked)
        decoder_input_val_marked = mark_captions(decoder_input_val)
        tokens_decoder_input_val = tokenizer.captions_to_tokens(
            decoder_input_val_marked)

        encoder_input_test_marked = mark_captions(encoder_input_test)
        tokens_encoder_input_test = tokenizer.captions_to_tokens(
            encoder_input_test_marked)
        decoder_input_test_marked = mark_captions(decoder_input_test)
        tokens_decoder_input_test = tokenizer.captions_to_tokens(
            decoder_input_test_marked)

        transfer_values_train = np.array(
            [transfer_values[filename] for filename in filenames_train])
        transfer_values_val = np.array(
            [transfer_values[filename] for filename in filenames_val])
        transfer_values_test = np.array(
            [transfer_values[filename] for filename in filenames_test])

        if emotion_training_mode == 'seq2seq':
            generator_train = seq2seq_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_train,
                tokens_encoder_input=tokens_encoder_input_train,
                tokens_decoder_input=tokens_decoder_input_train,
                with_transfer_values=with_transfer_value == 1)
            generator_val = seq2seq_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_val,
                tokens_encoder_input=tokens_encoder_input_val,
                tokens_decoder_input=tokens_decoder_input_val,
                with_transfer_values=with_transfer_value == 1)
            generator_test = seq2seq_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_test,
                tokens_encoder_input=tokens_encoder_input_test,
                tokens_decoder_input=tokens_decoder_input_test,
                with_transfer_values=with_transfer_value == 1)

        elif emotion_training_mode == 'stylenet':
            generator_train = stylenet_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_train,
                tokens=tokens_decoder_input_train,
                with_transfer_values=with_transfer_value == 1)
            generator_val = stylenet_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_val,
                tokens=tokens_decoder_input_val,
                with_transfer_values=with_transfer_value == 1)
            generator_test = stylenet_batch_generator(
                batch_size=batch_size,
                transfer_values=transfer_values_test,
                tokens=tokens_decoder_input_test,
                with_transfer_values=with_transfer_value == 1)

        if emotion_training_mode == 'seq2seq':

            total_num_captions_train = np.sum(
                [len(caption) for caption in encoder_input_train])
            total_num_captions_val = np.sum(
                [len(caption) for caption in encoder_input_val])
            total_num_captions_test = np.sum(
                [len(caption) for caption in encoder_input_test])
            train_steps = total_num_captions_train // batch_size
            val_steps = total_num_captions_val // batch_size
            test_steps = total_num_captions_test // batch_size

        elif emotion_training_mode == 'stylenet':

            total_num_captions_train = np.sum(
                [len(caption) for caption in decoder_input_train])
            total_num_captions_val = np.sum(
                [len(caption) for caption in decoder_input_val])
            total_num_captions_test = np.sum(
                [len(caption) for caption in decoder_input_test])
            train_steps = total_num_captions_train // batch_size
            val_steps = total_num_captions_val // batch_size
            test_steps = total_num_captions_test // batch_size

        print('train steps', train_steps)
        print('val steps', val_steps)
        print('test steps', test_steps)

        stylenet = StyleNet(
            injection_mode=injection_mode,
            num_words=num_words,
            trainable_model=False,
            mode=mode,
            state_size=state_size,
            embedding_size=embedding_size,
            factored_size=factored_size,
            lstm_layers=lstm_layers,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            dropout=dropout)

        try:
            stylenet.load(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        if emotion_training_mode == 'seq2seq':

            seq2seq = Seq2Seq(
                mode=mode,
                with_attention=with_attention == 1,
                injection_mode=injection_mode,
                num_words=num_words,
                state_size=state_size,
                embedding_size=embedding_size,
                factored_size=factored_size,
                encoder_lstm_layers=lstm_layers,
                decoder_lstm_layers=lstm_layers,
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                dropout=dropout)

            path_checkpoint += '.seq2seq'
            if load_model == 1:
                try:
                    seq2seq.load(path_checkpoint)
                except Exception as error:
                    print("Error trying to load checkpoint.")
                    print(error)
            seq2seq.set_encoder_weights(stylenet)

        if emotion_training_mode == 'seq2seq':
            callback_checkpoint = ModelCheckpoint(
                seq2seq,
                filepath=path_checkpoint,
                monitor='val_loss',
                verbose=1,
                save_best_only=True)
        elif emotion_training_mode == 'stylenet':
            callback_checkpoint = ModelCheckpoint(
                stylenet,
                filepath=path_checkpoint,
                monitor='val_loss',
                verbose=1,
                save_best_only=True)

        callback_tensorboard = TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=False)

        callbacks = [callback_checkpoint, callback_tensorboard]
        if early_stop > 0:
            callback_earystoping = EarlyStopping(
                monitor='val_loss',
                verbose=1,
                patience=early_stop,
                restore_best_weights=True)
            callbacks += [callback_earystoping]

        if emotion_training_mode == 'seq2seq':
            if with_transfer_value == 1:
                seq2seq.model.fit_generator(
                    generator=generator_train,
                    steps_per_epoch=train_steps,
                    epochs=epoch_num,
                    validation_data=generator_val,
                    validation_steps=val_steps,
                    callbacks=callbacks)

                scores = seq2seq.model.evaluate_generator(
                    generator=generator_test, steps=test_steps, verbose=1)
                print('test loss', scores)
            else:
                seq2seq.model_partial.fit_generator(
                    generator=generator_train,
                    steps_per_epoch=train_steps,
                    epochs=epoch_num,
                    validation_data=generator_val,
                    validation_steps=val_steps,
                    callbacks=callbacks)

                scores = seq2seq.model_partial.evaluate_generator(
                    generator=generator_test, steps=test_steps, verbose=1)
                print('test loss', scores)

            references = []
            predictions = []
            for filename, refs in zip(filenames_test, captions_test):
                _, output_text = generate_caption(
                    image_path=dataset_path + '/img/' + filename,
                    tokenizer=tokenizer,
                    stylenet=stylenet,
                    mode=mode,
                    k=beam_search,
                    seq2seq=seq2seq)
                predictions.append(output_text)
                references.append(refs)
            print(bleu_evaluator(references, predictions))

        elif emotion_training_mode == 'stylenet':
            if with_transfer_value == 1:
                stylenet.model_decoder.fit_generator(
                    generator=generator_train,
                    steps_per_epoch=train_steps,
                    epochs=epoch_num,
                    validation_data=generator_val,
                    validation_steps=val_steps,
                    callbacks=callbacks)
                scores = stylenet.model_decoder.evaluate_generator(
                    generator=generator_test, steps=test_steps, verbose=1)
                print('test loss', scores)
            else:
                stylenet.model_decoder_partial.fit_generator(
                    generator=generator_train,
                    steps_per_epoch=train_steps,
                    epochs=epoch_num,
                    validation_data=generator_val,
                    validation_steps=val_steps,
                    callbacks=callbacks)
                scores = stylenet.model_decoder_partial.evaluate_generator(
                    generator=generator_test, steps=test_steps, verbose=1)
                print('test loss', scores)

            references = []
            predictions = []
            for filename, refs in zip(filenames_test, captions_test):
                _, output_text = generate_caption(
                    image_path=dataset_path + '/img/' + filename,
                    tokenizer=tokenizer,
                    stylenet=stylenet,
                    k=beam_search,
                    mode=mode)
                predictions.append(output_text)
                references.append(refs)
            print(bleu_evaluator(references, predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='StyleNet Indonesia: Generate Image Commenting With Emotions'
    )
    parser.add_argument(
        '--emotion_training_mode',
        type=str,
        default='seq2seq',
        help='training mode for emotion. seq2seq or pure stylenet')
    parser.add_argument(
        '--beam_search',
        type=int,
        default=1,
        help='beam search for generating sequence')
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
        '--with_attention',
        type=int,
        default=1,
        help='use attention mechanism or not when training seq2seq emotion')
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
        '--early_stop',
        type=int,
        default=10,
        help='early stop if validation loss didn\'t improve')
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
        '--dropout',
        type=float,
        default=0.5,
        help='used for dropout rate for all specific layer')
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
