import numpy as np
import sys
# import time
import theano
from theano.ifelse import ifelse
import theano.tensor as T
# import sys
import pickle
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler, Normalizer
# from nltk.corpus import brown
from .mrnn_solver import RMSPROP
# import matplotlib.pyplot as plt
# import string
# import scipy.io
# import json
import copy
#import mkl
import spacy
from .mrnn_util import get_dictionary_a_to_b, make_drop_mask, build_len_masks, grad_clip, init_layer_xavier_1, \
    init_layer_k
from .mrnn_io import RNNDataProvider
sys.setrecursionlimit(50000)
#mkl.set_num_threads(6)

# sys.path.append("../LangModel")
#from mrnn_lm_modular import *


# finds the colosest word in the vocabulary using word2vec scores
class ClosestWordFinder:

    def __init__(self, w2i):
        self.nlp = spacy.load('en_core_web_lg')
        word_vecs = []
        word_to_i = []
        for w, i in list(w2i.items()):
            v = self.nlp.vocab[str(w)].vector
            word_vecs.append(v)
            word_to_i.append(i)
        word_vecs = np.array(word_vecs).T

        self.word_vecs = word_vecs
        self.word_to_i = word_to_i
        self.w2i = w2i

    def get_closest_word(self, chosen_word):
        if chosen_word in self.w2i:
            return self.w2i[chosen_word]

        v = self.nlp.vocab[str(chosen_word)].vector
        i = np.dot(v, self.word_vecs).argmax()
        i2w = dict([(w[1], w[0]) for w in list(self.w2i.items())])
        print(chosen_word, i2w[self.word_to_i[i]])
        return self.word_to_i[i]


# different domain adaption techniques to try
# an alpha value trading off the two RNNs
DA_FIXED_ALPHA = "DA_FIXED_ALPHA"
# try to keep the weights of the new matrix similar to the old one
DA_SIMILAR_PARAM = "DA_SIMILAR_PARAM"
# try to keep weights similar to old one and extra reg on imporant words
DA_SIMILAR_PARAM_2 = "DA_SIMILAR_PARAM_2"
# weight similar + extra reg + join two models
DA_SIMILAR_PARAM_3 = "DA_SIMILAR_PARAM_3"
# finetune old model with similar weights reg then join with new model using extra reg on important words and reg on \
# model mixing weights to increase usage of sentiment words
DA_SIMILAR_PARAM_SEPARATE = "DA_SIMILAR_PARAM_SEPARATE"
# join old model and new model with extra reg on important words and reg on model mixing weights to increase usage of
# sentiment words
DA_SUM = "DA_SUM"


class RNNModel:

    def __init__(self, conf_new=None):
        np.random.seed()

        conf = {}
        model = {}
        self.loaded_model = False

        conf['DOMAIN_ADAPT'] = DA_SUM
        #conf['DOMAIN_ADAPT'] = DA_FIXED_ALPHA
        #conf['DOMAIN_ADAPT'] = DA_SIMILAR_PARAM

        # when using DA_FIXED_ALPHA this is the weights of the two models
        conf['FIXED_ALPHA'] = 0.3
        # when using any of the models trying to keep weights similar this is the regulariaztion constant
        conf['SIMILAR_PARAM_REG'] = 10

        # which gradient decent method to use
        conf['GRAD_METHOD'] = RMSPROP

        # config for each of the gradient decent methods
        # momentum
        conf['DECAY_RATE'] = 0.99
        # fixed learning rate (or starting learning rate depending on method)
        conf['LEARNING_RATE'] = 0.001
        # the rho parameter for adadelta (no real need to tune this)
        conf['RHO_ADADELTA'] = 0.95

        # what regularisation to apply
        conf['L2_REG_CONST'] = 1e-8
        # dropout to apply to input Word Embeddings and input Image Embedding
        conf['DROP_INPUT_FRACTION'] = 0.5
        # droput to apply to output hidden layer
        conf['DROP_OUTPUT_FRACTION'] = 0.5
        conf['DROP_INPUT'] = True
        conf['DROP_OUTPUT'] = True
        # storing all unique droput masks is memory intensive only store dataset_size/DROP_MASK_SIZE_DIV instead
        conf['DROP_MASK_SIZE_DIV'] = 16

        # how much gradient clipping to apply (applied per element)
        conf['GRAD_CLIP_SIZE'] = 5

        # maximum length of a sentence
        # Note: actually this +1 because of start/end token
        conf['MAX_SENTENCE_LEN'] = 20

        # word must occur this many times in dataset to become part of vocab
        conf['MIN_WORD_FREQ'] = 5

        # number of sentences per mini-batch
        conf['batch_size_val'] = 200

        conf['DATASET'] = RNNDataProvider.FLK8

        # set layer sizes
        conf['emb_size'] = 256
        conf['lstm_hidden_size'] = 256
        conf['visual_size'] = 4096

        conf['SOFTMAX_OUT'] = True

        # join two RNNs (if JOINED_MODEL == True) then this is the second model
        conf['JOINED_MODEL'] = False
        conf['JOINED_LOSS_FUNCTION'] = False

        # initialize the ouput bias to the distribution of words in the vocabulary
        conf['INIT_OUTPUT_BIAS'] = True

        conf['TRAIN_SPLIT'] = RNNDataProvider.TRAIN
        conf['VAL_SPLIT'] = RNNDataProvider.VAL

        # apply normalization to the output of each layer
        # Note: didn't get any improvement from activating this
        conf['BATCH_NORM'] = False

        # with this probability select the true word as input at the next time step
        # otherwise select the generated word as input
        conf['SEMI_FORCED'] = 1

        # read the sentences in the reverse order
        conf['REVERSE_TEXT'] = False

        conf['SWITCHED'] = True
        conf['MAX_SWITCH_LEN'] = 3

        # 500
        conf['ATT_REG_CONST'] = 50
        # 500
        conf['LAMBDA_N'] = 0.25
        conf['LAMBDA_GAM'] = 0.25

        conf['params_bp_mask'] = {}

        # remember the important parameters for saving + SGD
        conf['param_names_saveable'] = [
            "wemb", "h0_hidden", "h0_cell", "wvm", "bmv", "w", "b", "w_lstm"
        ]
        if conf['BATCH_NORM']:
            conf['param_names_saveable'].append("gamma_h")
            conf['param_names_saveable'].append("beta_h")
        if conf['SWITCHED']:
            for layer in ["wemb_sw", "w_sw", "b_sw", "w_lstm_sw"]:
                conf['param_names_saveable'].append(layer)

        # set the parameters we want to train
        if conf['SWITCHED']:
            conf['param_names_trainable'] = [
                "wemb_sw", "w_sw", "b_sw", "w_lstm_sw", "wsenti"
            ]
        else:
            conf['param_names_trainable'] = [
                "wemb", "wvm", "bmv", "w", "b", "w_lstm"
            ]
            if conf['BATCH_NORM']:
                conf['param_names_trainable'].append("gamma_h")
                conf['param_names_trainable'].append("beta_h")

        self.model = model
        if conf_new is not None:
            for k, v in list(conf_new.items()):
                conf[k] = v

        self.conf = conf

    def set_as_joined_model(self, mm_rnn, lm_rnn):
        self.conf['JOINED_MODEL'] = True
        self.conf['SOFTMAX_OUT'] = False
        self.conf['JOINED_LOSS_FUNCTION'] = True

        self.mm_rnn = mm_rnn
        self.lm_rnn = lm_rnn

        # build conversion from lm dictionary to mm dictionary
        mm2lm_map, mm2lm, mm2lm_mask, _ = get_dictionary_a_to_b(
            mm_rnn.model['w2i'], mm_rnn.model['i2w'], lm_rnn.model['w2i'],
            lm_rnn.model['i2w'])

        self.mm2lm = mm2lm
        self.mm2lm_map = mm2lm_map

        # convert the output of the lm to match the order of the mm (zero out when doesn't exist)
        m2l = theano.shared(mm2lm, name="mm2lm", borrow=True)
        m2l_m = theano.shared(mm2lm_mask, name="mm2lm_mask", borrow=True)
        self.lm_new_s = lm_rnn.new_s[:, m2l] * m2l_m
        self.lm_new_s = self.lm_new_s / \
            self.lm_new_s.sum(axis=1, keepdims=True)

    def save_model(self, filename):
        # get the model parameters and save them
        params_saved = dict([(p, getattr(self, p).get_value(borrow=True))
                             for p in self.conf['param_names_saveable']])
        self.model['params_saved'] = params_saved
        self.model['hist_grad'] = {}
        self.model['delta_grad'] = {}
        for p in self.conf['param_names_saveable']:
            if p in self.conf['param_names_trainable']:
                idx = self.conf['param_names_trainable'].index(p)
                self.model['hist_grad'][p] = self.hist_grad[idx].get_value(
                    borrow=True)
                self.model['delta_grad'][p] = self.delta_grad[idx].get_value(
                    borrow=True)

        pickle.dump((self.conf, self.model), open(filename, "wb"), protocol=2)

    def build_shared_layers(self, layers):
        for name, data in list(layers.items()):
            setattr(self, name, theano.shared(name=name,
                                              value=data,
                                              borrow=True))

    def load_model(self, filename, conf=None, load_solver_params=True):
        self.__init__()
        self.loaded_model = True

        conf_new, self.model = pickle.load(open(filename, "rb"), encoding='latin1')
        for k, v in list(conf_new.items()):
            self.conf[k] = v
        if conf is not None:
            for k, v in list(conf.items()):
                self.conf[k] = v

        shared_layers = self.model['params_saved']
        if self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_SEPARATE:
            shared_layers_new = []
            subst = dict([("wemb_sw", "wemb"), ("w_sw", "w"), ("b_sw", "b"),
                          ("w_lstm_sw", "w_lstm"), ("wvm_sw", "wvm"),
                          ("bmv_sw", "bmv")])
            for name, data in shared_layers:
                if name in subst:
                    shared_layers_new.append((subst[name], data))

        self.build_shared_layers(shared_layers)

        if load_solver_params:
            self.hist_grad = []
            self.delta_grad = []
            for i, p in enumerate(self.conf['param_names_trainable']):
                if p in self.model['hist_grad']:
                    if hasattr(self, p):
                        v = self.model['hist_grad'][p]
                        self.hist_grad.append(
                            theano.shared(name="hist_grad[%d]" % i,
                                          value=v,
                                          borrow=True))
                else:
                    if hasattr(self, p):
                        v = np.zeros_like(
                            getattr(self, p).get_value(borrow=True))
                        self.hist_grad.append(
                            theano.shared(name="hist_grad[%d]" % i,
                                          value=v,
                                          borrow=True))
                if p in self.model['delta_grad']:
                    if hasattr(self, p):
                        v = self.model['delta_grad'][p]
                        self.delta_grad.append(
                            theano.shared(name="delta_grad[%d]" % i,
                                          value=v,
                                          borrow=True))
                else:
                    if hasattr(self, p):
                        v = np.zeros_like(
                            getattr(self, p).get_value(borrow=True))
                        self.delta_grad.append(
                            theano.shared(name="delta_grad[%d]" % i,
                                          value=v,
                                          borrow=True))

    # dropout masks that do nothing
    def make_drop_masks_identity(self, n_instances):
        mX = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN'] + 1,
                     self.conf['emb_size']),
                    dtype=theano.config.floatX)
        mY = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN'] + 1,
                     self.conf['lstm_hidden_size']),
                    dtype=theano.config.floatX)
        return mX, mY

    def make_new_train_drop_masks(self):
        div = self.conf['DROP_MASK_SIZE_DIV']
        num_drop_masks = self.X_train.shape[0] / div + div
        if num_drop_masks < self.conf['batch_size_val']:
            num_drop_masks = self.conf['batch_size_val']
        new_X_drop = make_drop_mask(num_drop_masks,
                                    self.conf['MAX_SENTENCE_LEN'] + 1,
                                    self.conf['emb_size'],
                                    self.conf['DROP_INPUT_FRACTION'])
        new_Y_drop = make_drop_mask(num_drop_masks,
                                    self.conf['MAX_SENTENCE_LEN'] + 1,
                                    self.conf['lstm_hidden_size'],
                                    self.conf['DROP_OUTPUT_FRACTION'])
        if hasattr(self, 'X_sh_train_drop'):
            self.X_train_drop = new_X_drop
            self.X_sh_train_drop.set_value(new_X_drop)
        else:
            self.X_train_drop = new_X_drop

        if hasattr(self, 'Y_sh_train_drop'):
            self.Y_train_drop = new_Y_drop
            self.Y_sh_train_drop.set_value(new_Y_drop)
        else:
            self.Y_train_drop = new_Y_drop

    def setup_dataprovider(self, load_vocab=True):
        if not hasattr(self, 'dp'):
            self.dp = RNNDataProvider(self.conf['DATASET'],
                                      self.conf['REVERSE_TEXT'])
        if not self.dp.loaded:
            # read the sentences
            self.dp.read_dataset()
            # read the contexts
            self.dp.read_context()
            if not load_vocab:
                self.dp.w2i = self.model['w2i']
                self.dp.i2w = self.model['i2w']
            else:
                self.dp.build_vocab(self.conf['MIN_WORD_FREQ'])
                self.model['i2w'] = self.dp.i2w
                self.model['w2i'] = self.dp.w2i

    # change the vocabulary used to define the sentences in X
    def convert_X_to_lm_vocab(self, X):
        X_lm = np.zeros_like(X)
        last_tok = 1
        for r in range(X_lm.shape[0]):
            for c in range(X_lm.shape[1]):
                if X[r, c] in self.mm2lm_map:
                    X_lm[r, c] = self.mm2lm_map[X[r, c]]
                    if X_lm[r, c] != 0:
                        last_tok = X_lm[r, c]
                else:
                    X_lm[r, c] = last_tok
            # print X_lm[r]
            # if np.random.randint(15) == 1:
            #    sys.exit(0)
        return X_lm

    def fill_extra_SW(self, SW):
        return
        for r in range(SW.shape[0]):
            change_pos = []
            for c in range(0, SW.shape[1]):
                if SW[r, c] == 1:
                    change_pos.append(c)
            for c in change_pos:
                SW[r, c:c + self.conf['MAX_SWITCH_LEN']] = 1

    def load_training_dataset(self, merge_vocab=False):
        # load the dataset into memory
        # also construct a new vocabulary (saved in self.model['w2i'])
        if merge_vocab:
            self.setup_dataprovider(load_vocab=False)
            i2w_old = self.dp.i2w
            w2i_old = self.dp.w2i
            self.dp.build_vocab(self.conf['MIN_WORD_FREQ'])
            # i2w_new = self.dp.i2w
            w2i_new = self.dp.w2i
            print("Old len:", len(i2w_old))

            new_words = set(w2i_new.keys()) - set(w2i_old.keys())
            cur_idx = np.amax(list(i2w_old.keys())) + 1
            print(len(new_words))

            self.conf['added_words'] = []

            for w in new_words:
                i2w_old[cur_idx] = w
                w2i_old[w] = cur_idx
                self.conf['added_words'].append((w, cur_idx))
                cur_idx += 1

            self.model['i2w'] = i2w_old
            self.model['w2i'] = w2i_old
            self.dp.i2w = i2w_old
            self.dp.w2i = w2i_old
        else:
            self.setup_dataprovider(load_vocab=False)

        # get the training dataset split
        if self.conf['SWITCHED']:
            self.X_train, self.Xlen_train, self.V_train, _, self.SW_train, self.senti_train = self.dp.get_data_split(
                data_split=self.conf['TRAIN_SPLIT'],
                randomize=True,
                pad_len=self.conf['MAX_SENTENCE_LEN'],
                anp_switch=True)
        else:
            self.X_train, self.Xlen_train, self.V_train, _ = self.dp.get_data_split(
                data_split=self.conf['TRAIN_SPLIT'],
                randomize=True,
                pad_len=self.conf['MAX_SENTENCE_LEN'])
        self.X_train_mask = build_len_masks(self.Xlen_train + 1,
                                            self.conf['MAX_SENTENCE_LEN'] + 1)
        self.num_train_examples = self.X_train.shape[0]

        # expand out the ones in the switch array
        # self.fill_extra_SW(self.SW_train)

        for i in range(10):
            print(self.SW_train[i])
            print(self.senti_train[i])
            for w, sw in zip(self.X_train[i], self.SW_train[i]):
                print("%s_%d" % (self.model['i2w'][w], sw), end=' ')
            print("\n")
        # sys.exit(0)

        if self.conf['JOINED_MODEL']:
            # map from the mm dictionary into the lm dictionary
            # if no dictionary entry repeat the last one (dont ever repeat START or STOP tokens)
            self.X_train_lm = self.convert_X_to_lm_vocab(self.X_train)

        # build the dropout masks
        self.make_new_train_drop_masks()

        self.conf['vocab_size'] = len(self.model['i2w'])
        self.conf['output_size'] = len(self.model['i2w'])

    def load_val_dataset(self):
        self.setup_dataprovider(load_vocab=False)
        if self.conf['SWITCHED']:
            tmp = self.dp.get_data_split(data_split=self.conf['VAL_SPLIT'],
                                         randomize=False,
                                         pad_len=self.conf['MAX_SENTENCE_LEN'],
                                         anp_switch=True)
            self.X_valid, self.Xlen_valid, self.V_valid, self.Id_valid, self.SW_valid, self.senti_valid = tmp
        else:
            tmp = self.dp.get_data_split(data_split=self.conf['VAL_SPLIT'],
                                         randomize=False,
                                         pad_len=self.conf['MAX_SENTENCE_LEN'])
            self.X_valid, self.Xlen_valid, self.V_valid, self.Id_valid = tmp
        self.X_valid_mask = build_len_masks(self.Xlen_valid + 1,
                                            self.conf['MAX_SENTENCE_LEN'] + 1)

        # expand out the ones in the switch array
        # self.fill_extra_SW(self.SW_valid)

    def build_model_core(self):

        # gradient clipping function
        self.clipg = lambda x: grad_clip(x, -self.conf['GRAD_CLIP_SIZE'], self.
                                         conf['GRAD_CLIP_SIZE'])

        shared_layers = {}

        # do sergery on wemb, w and b layers
        if hasattr(self, 'wemb'):
            wemb_vals = self.wemb.get_value(borrow=False)
            w_vals = self.w.get_value(borrow=False)
            b_vals = self.b.get_value(borrow=False)
            if hasattr(self, 'dp') and wemb_vals.shape[0] != len(self.dp.i2w):
                old_w2i = copy.deepcopy(self.dp.w2i)
                for w, i in self.conf['added_words']:
                    del old_w2i[w]
                cwf = ClosestWordFinder(old_w2i)

                tmp = [v[1] for v in self.conf['added_words']]
                num_new_rows = np.amax(tmp) - wemb_vals.shape[0] + 1
                wemb_vals = np.vstack([
                    wemb_vals,
                    np.zeros((num_new_rows, wemb_vals.shape[1]),
                             dtype=theano.config.floatX)
                ])
                # print w_vals.shape
                # print b_vals.shape
                w_vals = np.hstack([
                    w_vals,
                    np.zeros((w_vals.shape[0], num_new_rows),
                             dtype=theano.config.floatX)
                ])
                b_vals = np.concatenate([
                    b_vals,
                    np.zeros((num_new_rows,), dtype=theano.config.floatX)
                ])
                for w, i in self.conf['added_words']:
                    ci = cwf.get_closest_word(w)
                    wemb_vals[i, :] = wemb_vals[ci, :]
                    w_vals[:, i] = w_vals[:, ci]
                    b_vals[i] = b_vals[ci]
                # print b_vals.shape
                # print w_vals.shape
                # print b_vals
                self.wemb.set_value(wemb_vals)
                self.w.set_value(w_vals)
                self.b.set_value(b_vals)

            #shared_layers['wemb'] = wemb_val
        # sys.exit(0)

        if self.conf['SWITCHED']:
            if not hasattr(self, 'wemb_sw'):
                shared_layers['wemb_sw'] = self.wemb.get_value(borrow=False)
            if not hasattr(self, 'w_lstm_sw'):
                shared_layers['w_lstm_sw'] = self.w_lstm.get_value(
                    borrow=False)
            if not hasattr(self, 'w_sw'):
                shared_layers['w_sw'] = self.w.get_value(borrow=False)
            if not hasattr(self, 'b_sw'):
                shared_layers['b_sw'] = self.b.get_value(borrow=False)
            if not hasattr(self, 'wvm_sw'):
                shared_layers['wvm_sw'] = self.wvm.get_value(borrow=False)
            if not hasattr(self, 'bmv_sw'):
                shared_layers['bmv_sw'] = self.bmv.get_value(borrow=False)

            if not hasattr(self, 'wemb_sw2'):
                shared_layers['wemb_sw2'] = self.wemb.get_value(borrow=False)
            if not hasattr(self, 'w_lstm_sw2'):
                shared_layers['w_lstm_sw2'] = self.w_lstm.get_value(
                    borrow=False)
            if not hasattr(self, 'w_sw2'):
                shared_layers['w_sw2'] = self.w.get_value(borrow=False)
            if not hasattr(self, 'b_sw2'):
                shared_layers['b_sw2'] = self.b.get_value(borrow=False)

        if self.conf['BATCH_NORM']:
            if not hasattr(self, 'gamma_h'):
                gamma_h_val = np.ones((self.conf['lstm_hidden_size'] * 2,),
                                      dtype=theano.config.floatX)
                shared_layers['gamma_h'] = gamma_h_val
            if not hasattr(self, 'beta_h'):
                beta_h_val = np.zeros((self.conf['lstm_hidden_size'] * 2,),
                                      dtype=theano.config.floatX)
                shared_layers['beta_h'] = beta_h_val

        if not hasattr(self, 'att_w'):
            att_w_val = init_layer_xavier_1(self.conf['lstm_hidden_size'] * 2,
                                            1,
                                            scale=0.5)
            shared_layers['att_w'] = att_w_val

        if not hasattr(self, 'att_wv'):
            att_wv_val = init_layer_xavier_1(self.conf['emb_size'],
                                             1,
                                             scale=1 / 160.0)
            shared_layers['att_wv'] = att_wv_val

        if not hasattr(self, 'att_w2'):
            att_w2_val = init_layer_k(self.conf['lstm_hidden_size'], 1)
            shared_layers['att_w2'] = att_w2_val

        if not hasattr(self, 'wsenti'):
            wsenti_val = init_layer_k(self.conf['lstm_hidden_size'], 1)
            shared_layers['wsenti'] = wsenti_val

        if not hasattr(self, 'wsenti2'):
            wsenti2_val = init_layer_k(self.conf['lstm_hidden_size'], 1)
            shared_layers['wsenti2'] = wsenti2_val

        if not hasattr(self, 'att_b'):
            att_b_val = np.zeros((1,), dtype=theano.config.floatX)
            shared_layers['att_b'] = att_b_val

        # set the default network weights
        if not hasattr(self, 'wemb'):
            wemb_val = init_layer_k(
                self.conf['vocab_size'], self.conf['emb_size'])
            shared_layers['wemb'] = wemb_val

        if not hasattr(self, 'h0_hidden'):
            h0_hidden_val = np.zeros(
                (self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
            shared_layers['h0_hidden'] = h0_hidden_val

        if not hasattr(self, 'h0_cell'):
            h0_cell_val = np.zeros(
                (self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
            shared_layers['h0_cell'] = h0_cell_val

        # mapping from visual space to word space
        if not hasattr(self, 'wvm'):
            wvm_val = init_layer_k(
                self.conf['visual_size'], self.conf['emb_size'])
            shared_layers['wvm'] = wvm_val

        if not hasattr(self, 'bmv'):
            bmv_val = np.zeros(
                (self.conf['emb_size'],), dtype=theano.config.floatX)
            shared_layers['bmv'] = bmv_val

        # LSTM layer parameters
        if not hasattr(self, 'w_lstm'):
            lstm_hidden_size = self.conf['lstm_hidden_size']
            w_lstm_val = init_layer_k(lstm_hidden_size * 2,
                                      lstm_hidden_size * 4)
            shared_layers['w_lstm'] = w_lstm_val

        # mapping from RNN hidden output to vocabulary
        if not hasattr(self, 'w'):
            w_val = init_layer_k(
                self.conf['lstm_hidden_size'], self.conf['output_size'])
            shared_layers['w'] = w_val

        if not hasattr(self, 'b'):
            b_val = np.zeros(
                (self.conf['output_size'],), dtype=theano.config.floatX)
            if self.conf["INIT_OUTPUT_BIAS"]:
                # set the bias on the last layer to be the log prob of each of the words in the vocab
                wcount = 0
                w2i = self.dp.w2i
                w2c = self.dp.get_word_counts(RNNDataProvider.TRAIN)
                for w in w2i:
                    if w in w2c:
                        wcount += w2c[w]
                wcount += self.X_train.shape[0]
                tmp = np.log(self.X_train.shape[0] / float(wcount))
                b_val[w2i[RNNDataProvider.STOP_TOKEN]] = tmp
                for w in w2i:
                    if w in w2c:
                        tmp = np.log(w2c[w] / float(wcount))
                        b_val[w2i[w]] = tmp
                b_val -= np.max(b_val[1:])
            shared_layers['b'] = b_val

        self.build_shared_layers(shared_layers)

        # input variables for training
        self.x = T.imatrix("x")
        self.v = T.matrix("v")
        self.senti = T.vector("senti")
        self.is_switched_seq = T.matrix("is_switched_seq")
        self.xlen = T.matrix("xlen")
        self.neg_sgd = T.scalar("neg_sgd")

        # input variables for generation
        self.v_single = T.vector("v")
        self.nstep = T.iscalar("nstep")

        # the dropout masks
        # drop the input
        self.x_drop = T.tensor3("x_drop")
        # drop the output
        self.y_drop = T.tensor3("y_drop")

        self.forced_word = T.imatrix("forced_word")

        # hidden layer ouput
        h_tm1 = T.vector("h_tm1")
        # word indexes
        word_t = T.ivector("word_t")
        # visual information
        v_i = T.vector("v")

        # Generates the next word based on the: previous true word, hidden state & visual features
        # inputs: hiddent_layer, last_predicted word, visual features
        def recurrance(word_t, x_drop_slice, hh_drop_slice, use_v, is_switched,
                       h_tm1_hidden, h_tm1_cell, v_i, senti):

            def rec_part(word_t, x_drop_slice, hh_drop_slice, use_v, h_tm1_hidden,
                         h_tm1_cell, v_i, senti, c_wemb, c_w, c_b, c_w_lstm, c_wvm,
                         c_bmv):
                # get the word embedding matrix or the context information
                x_t = ifelse(T.eq(use_v, 1), T.dot(
                    v_i, c_wvm) + c_bmv, c_wemb[word_t])
                #x_t = theano.printing.Print("x_t", ["shape"])(x_t)
                #senti = theano.printing.Print("senti", ["shape"])(senti)

                # if we are not doing minibatch training
                if x_t.ndim == 1:
                    x_t = x_t.reshape((1, x_t.shape[0]))
                if h_tm1_hidden.ndim == 1:
                    h_tm1_hidden = h_tm1_hidden.reshape(
                        (1, h_tm1_hidden.shape[0]))
                    #h_tm1_hidden = theano.printing.Print("h_tm1_hidden", ["shape"])(h_tm1_hidden)

                #senti = theano.printing.Print("senti")(senti)
                #sm = T.dot(self.wsenti, senti.reshape((1,senti.shape[0])))
                #sm2 = T.dot(self.wsenti2, (1.0 - senti.reshape((1,senti.shape[0]))))
                #sm_out = ifelse(T.le(senti[0], -0.5), x_t.T * T.ones_like(sm), sm + sm2)
                #sm = theano.printing.Print("sm", ["shape"])(sm)

                # x_t_new = T.cast(1.0-use_v, dtype=theano.config.floatX) * x_t + \
                # T.cast(use_v, dtype=theano.config.floatX) * sm.T

                #x_t = ifelse(T.eq(use_v, 1), sm_out.T, x_t * T.ones_like(sm.T))
                #x_t = sm_out.T

                #h_tm1_hidden = theano.printing.Print("h_tm1_hidden", ["shape"])(h_tm1_hidden)

                # dropout on the input embddings
                if self.conf['DROP_INPUT']:
                    x_t *= x_drop_slice

                # clip the gradients so they dont get too large
                h_tm1_hidden_clip = self.clipg(h_tm1_hidden)

                #x_t = theano.printing.Print("x_t", ["shape"])(x_t)
                #h_tm1_hidden_clip = theano.printing.Print("h_tm1_hidden_clip", ["shape"])(h_tm1_hidden_clip)
                in_state = T.concatenate([x_t, h_tm1_hidden_clip], axis=1)

                #in_state = theano.printing.Print("in_state", ['shape'])(in_state)
                #c_w_lstm = theano.printing.Print("c_w_lstm", ['shape'])(c_w_lstm)

                if False and self.conf['BATCH_NORM']:
                    mu = T.mean(in_state, axis=0, keepdims=True)
                    var = T.var(in_state, axis=0, keepdims=True)
                    normed_is = (in_state - mu) / T.sqrt(
                        var + T.constant(1e-10, dtype=theano.config.floatX))
                    in_state = self.gamma_h * in_state + self.beta_h

                # calculate 8 dot products in one go
                dot_out = T.dot(in_state, c_w_lstm)

                lstm_hidden_size = self.conf['lstm_hidden_size']
                # input gate
                ig = T.nnet.sigmoid(dot_out[:, :lstm_hidden_size])
                # forget gate
                fg = T.nnet.sigmoid(
                    dot_out[:, lstm_hidden_size:lstm_hidden_size * 2])
                # output gate
                og = T.nnet.sigmoid(dot_out[:, lstm_hidden_size * 2:lstm_hidden_size *
                                            3])

                # cell memory
                cc = fg * h_tm1_cell + ig * \
                    T.tanh(dot_out[:, lstm_hidden_size * 3:])

                # hidden state
                hh = og * cc

                # * sm.T
                hh = hh

                # drop the output state
                if self.conf['DROP_OUTPUT']:
                    hh = hh * hh_drop_slice

                # the distribution over output words
                if self.conf['SOFTMAX_OUT']:
                    #out_l1 = T.dot(hh, c_w) + c_b
                    #out_l2 = T.dot(hh, c_w2) + c_b2
                    #senti = theano.printing.Print("senti")(senti)
                    #out_layer = senti * out_l1.T + (1.0 - senti) * out_l2.T
                    #s_t1 = T.nnet.softmax(out_layer.T)
                    # * T.ones_like(s_t1)
                    s_t = T.nnet.softmax(T.dot(hh, c_w) + c_b)
                    #s_t = ifelse(T.le(senti[0], -0.5), s_t2, s_t1)
                else:
                    s_t = T.nnet.sigmoid(T.dot(hh, c_w) + c_b)

                # if we are not doing minibatch training
                # if word_t.ndim == 0:
                #    hh = hh[0]
                #    cc = cc[0]

                return [hh, cc, s_t]

            if h_tm1_hidden.ndim == 1:
                h_tm1_hidden = h_tm1_hidden.reshape((1, h_tm1_hidden.shape[0]))
            if h_tm1_cell.ndim == 1:
                h_tm1_cell = h_tm1_cell.reshape((1, h_tm1_cell.shape[0]))

            last_hh_orig = h_tm1_hidden[:h_tm1_hidden.shape[0] // 2, :]
            last_hh_new = h_tm1_hidden[h_tm1_hidden.shape[0] // 2:, :]
            last_cc_orig = h_tm1_cell[:h_tm1_cell.shape[0] // 2, :]
            last_cc_new = h_tm1_cell[h_tm1_cell.shape[0] // 2:, :]
            #last_hh_orig = theano.printing.Print("last_hh_orig", ["shape"])(last_hh_orig)
            #last_cc_orig = theano.printing.Print("last_cc_orig", ["shape"])(last_cc_orig)

            hh_orig, cc_orig, s_t_orig = rec_part(word_t,
                                                  T.ones_like(x_drop_slice),
                                                  T.ones_like(hh_drop_slice),
                                                  use_v, last_hh_orig,
                                                  last_cc_orig, v_i,
                                                  T.ones_like(senti) * -1.0,
                                                  self.wemb, self.w, self.b,
                                                  self.w_lstm, self.wvm,
                                                  self.bmv)

            hh_new, cc_new, s_t_new = rec_part(word_t, x_drop_slice,
                                               hh_drop_slice, use_v,
                                               last_hh_new, last_cc_new, v_i,
                                               senti, self.wemb_sw, self.w_sw,
                                               self.b_sw, self.w_lstm_sw,
                                               self.wvm_sw, self.bmv_sw)

            # hh_new2, cc_new2, s_t_new2 = rec_part(word_t, x_drop_slice, hh_drop_slice, use_v, last_hh_new, last_cc_new,
            # v_i, senti, self.wemb_sw2, self.w_sw2, self.b_sw2, self.w_lstm_sw2, self.wvm, self.bmv)
            #senti = theano.printing.Print("senti")(senti)
            #hh_new = theano.printing.Print("hh_new")(hh_new)
            #hh_new2 = theano.printing.Print("hh_new2")(hh_new)

            #hh_new = (senti * hh_new.T + (1.0 - senti) * hh_new2.T).T
            #cc_new = (senti * cc_new.T + (1.0 - senti) * cc_new2.T).T
            #s_t_new = (senti * s_t_new.T + (1.0 - senti) * s_t_new2.T).T

            #hh_new = theano.printing.Print("hh_new_pre")(hh_new)
            # hh_new = (senti * hh_new.T).T #+ #(1.0 - senti) * hh_new2.T).T
            # cc_new = (senti * cc_new.T).T #+ #(1.0 - senti) * cc_new2.T).T
            # s_t_new = (senti * s_t_new.T).T #+ #(1.0 - senti) * s_t_new2.T).T
            #hh_new = theano.printing.Print("hh_new_out")(hh_new)
            # hh_new = hh_orig cc_new = cc_orig s_t_new = s_t_orig

            unit = np.array(1, dtype=theano.config.floatX)
            if is_switched.ndim != 0:
                is_switched_col = is_switched.dimshuffle((0, 'x'))
                #is_switched_col = T.printing.Print("is_switched")(is_switched_col)
            else:
                is_switched_col = is_switched.dimshuffle(('x'))

            #att_wall = senti.T * self.att_w + (1.0-senti.T) * self.att_w2
            # gradient clipping function
            #self.clipg2 = lambda x: grad_clip(x, -0.0001, 0.0001)
            #att_wv_t = theano.printing.Print("self.att_wv", ["sum", "mean"])(self.att_wv + T.zeros_like(self.att_wv))
            #vs_for_att = T.dot(T.dot(v_i, self.wvm) * x_drop_slice, self.clipg2(self.att_wv))
            #vs_for_att = theano.printing.Print("vs_for_att", ["sum", "mean"])(vs_for_att)
            #T.dot(v_i, c_wvm) + c_bmv
            att_new = T.nnet.sigmoid(
                T.dot(T.concatenate([hh_orig, hh_new], axis=1), self.att_w) + self.att_b).T[0]
            #att_new_b = T.nnet.sigmoid(T.dot(hh_new, self.att_w2) + self.att_b).T[0]
            #att_new = senti.T * att_new_a + (1.0 - senti.T) * att_new_b
            #att_new = T.nnet.sigmoid(T.dot(hh_orig, att_wall) + self.att_b).T[0]
            #att_new = theano.printing.Print("att_new", ["shape"])(att_new)
            # if word_t.ndim != 0:
            att_new = att_new.dimshuffle((0, 'x'))
            #att_new = theano.printing.Print("att_new")(att_new)
            #att_new = theano.printing.Print("att_new", ["shape"])(att_new)
            #s_t_new = theano.printing.Print("s_t_new", ["shape"])(s_t_new)
            #hh_out = T.cast((unit-is_switched_col) * hh_orig + is_switched_col * hh_new, 'float32')
            #cc_out = T.cast((unit-is_switched_col) * cc_orig + is_switched_col * cc_new, 'float32')
            #s_t_out = T.cast((unit-is_switched_col) * s_t_orig + is_switched_col * s_t_new, 'float32')

            #hh_out = T.cast((unit-att_new) * hh_orig + att_new * hh_new, 'float32')
            #cc_out = T.cast((unit-att_new) * cc_orig + att_new * cc_new, 'float32')
            #s_t_out = T.cast((unit-att_new) * s_t_orig + att_new * s_t_new, 'float32')
            if self.conf['DOMAIN_ADAPT'] == DA_SUM:
                s_t_out = ifelse(T.le(senti[0], -0.5), s_t_orig, T.cast((unit - att_new) * s_t_orig + att_new * s_t_new,
                                                                        'float32'))
                # s_t_out = ifelse(T.le(senti[0],-0.5), s_t_orig,
                # T.cast(s_t_orig * (1.0 + (s_t_new - s_t_orig) / att_new), 'float32'))
                #s_t_out = theano.printing.Print("s_t_out", ["shape"])(s_t_out)
                #s_t_out = s_t_out / T.sum(s_t_out, axis=1, keepdims=True)
            elif self.conf['DOMAIN_ADAPT'] == DA_FIXED_ALPHA:
                tmp = (1.0 - self.conf['FIXED_ALPHA']) * s_t_orig
                tmp2 = self.conf['FIXED_ALPHA'] * s_t_new
                s_t_out = ifelse(
                    T.le(senti[0], -0.5), s_t_orig, T.cast(tmp + tmp2, 'float32'))
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM:
                s_t_out = ifelse(T.le(senti[0], -0.5), s_t_orig, s_t_new)
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_2:
                s_t_out = ifelse(T.le(senti[0], -0.5), s_t_orig, s_t_new)
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_3:
                tmp = (unit - att_new) * s_t_orig
                tmp2 = att_new * s_t_new
                s_t_out = ifelse(
                    T.le(senti[0], -0.5), s_t_orig, T.cast(tmp + tmp2, 'float32'))

            #hh_orig = theano.printing.Print("hh_orig", ["shape"])(hh_orig)
            #hh_new = theano.printing.Print("hh_new", ["shape"])(hh_new)
            hh_out = T.concatenate([hh_orig, hh_new])
            cc_out = T.concatenate([cc_orig, cc_new])

            # return [hh_orig, cc_orig, s_t_orig]
            return [hh_out, cc_out, s_t_out, T.log(att_new), T.log(1.0 - att_new)]
            # return [hh_out, cc_out, s_t_new, att_new]

        # Generates the next word by feeding the old word as input
        # inputs: hiddent_layer, last_predicted word, visual features
        def recurrance_word_feedback(h_tm1_hidden, h_tm1_cell, word_t, use_visual, v_i,
                                     senti):
            x_drop_val = T.ones(
                (self.conf['emb_size'],), dtype=theano.config.floatX)
            y_drop_val = T.ones((self.conf['lstm_hidden_size'],),
                                dtype=theano.config.floatX)
            #is_switched = np.zeros((word_t.shape[0],), dtype=theano.config.floatX)
            is_switched = T.zeros_like(word_t)
            [hh, cc, s_t, att, att2] = recurrance(word_t, x_drop_val, y_drop_val, use_visual, is_switched,
                                                  h_tm1_hidden, h_tm1_cell, v_i, senti)

            # the predicted word
            w_idx = T.cast(T.argmax(s_t, axis=1), dtype='int32')[0]
            print(hh.ndim, h_tm1_hidden.ndim)

            return [hh, cc, s_t[0], w_idx, T.zeros((0,), dtype='int32')[0], att]

        def recurrance_partial_word_feedback(word_t_real, x_drop_val, y_drop_val,
                                             use_visual, forced_word, is_switched,
                                             h_tm1_hidden, h_tm1_cell, word_t_pred, v_i,
                                             senti):
            word_last = T.switch(forced_word, word_t_real, word_t_pred)
            [hh, cc, s_t, att, att2] = recurrance(word_last, x_drop_val, y_drop_val, use_visual,
                                                  is_switched, h_tm1_hidden, h_tm1_cell, v_i, senti)

            # the predicted word
            w_idx = T.cast(T.argmax(s_t, axis=1), dtype='int32')
            return [hh, cc, s_t, w_idx]

        # build the teacher forcing loop
        use_visual_info = T.concatenate([T.ones((1,), dtype=np.int32), T.zeros((self.conf['MAX_SENTENCE_LEN'],),
                                                                               dtype=np.int32)])
        h0_hidden_matrix = self.h0_hidden * \
            T.ones((self.x.shape[0] * 2, self.h0_hidden.shape[0]))
        h0_cell_matrix = self.h0_cell * \
            T.ones((self.x.shape[0] * 2, self.h0_cell.shape[0]))
        x_adj = T.concatenate(
            [T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype), self.x.T])
        y_adj = T.concatenate(
            [self.x.T, T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype)])
        is_switched_seq = self.is_switched_seq.T
        #is_switched_seq = theano.printing.Print("is_switched")(is_switched_seq)
        [_, _, s, self.att, self.att2], _ = theano.scan(fn=recurrance,
                                                        sequences=[x_adj, self.x_drop.dimshuffle((1, 0, 2)), self.y_drop.dimshuffle(
                                                            (1, 0, 2)), use_visual_info, is_switched_seq],
                                                        n_steps=self.conf['MAX_SENTENCE_LEN']+1,
                                                        non_sequences=[
                                                            self.v, self.senti],
                                                        outputs_info=[h0_hidden_matrix, h0_cell_matrix, None, None, None])

        # build the semi-forced loop
        [_, _, s_semi, _], _ = theano.scan(fn=recurrance_partial_word_feedback,
                                           sequences=[x_adj, self.x_drop.dimshuffle((1, 0, 2)), self.y_drop.dimshuffle((1, 0, 2)),
                                                      use_visual_info, self.forced_word[:, :self.x.shape[0]], is_switched_seq],
                                           n_steps=self.conf['MAX_SENTENCE_LEN']+1,
                                           non_sequences=[self.v, self.senti],
                                           outputs_info=[h0_hidden_matrix, h0_cell_matrix, None, T.zeros((self.x.shape[0],), dtype=np.int32)])

        # build the un-forced loop
        h0_hidden_matrix = self.h0_hidden * \
            T.ones((2, self.h0_hidden.shape[0]))
        h0_cell_matrix = self.h0_cell * T.ones((2, self.h0_cell.shape[0]))
        print(h0_hidden_matrix.ndim)

        [_, _, _, self.wout_fb, _, self.att_gen], _ = theano.scan(fn=recurrance_word_feedback,
                                                                  non_sequences=[
                                                                      self.v_single, self.senti],
                                                                  outputs_info=[h0_hidden_matrix, h0_cell_matrix, None, np.array(
                                                                      0, dtype=np.int32), T.ones((1,), dtype=np.int32)[0], None],
                                                                  n_steps=self.nstep)

        if self.conf['SEMI_FORCED'] < 1:
            s = s_semi

        #self.att = theano.printing.Print("att", ["shape"])(self.att)
        self.att = self.att[:, :, 0]
        self.att2 = self.att2[:, :, 0]

        self.new_s = s.reshape((s.shape[0] * s.shape[1], s.shape[2]))
        softmax_out = self.build_loss_function(self.new_s, y_adj)
        self.softmax_out = softmax_out

        #self.att = T.sum(T.clip(self.att - 0.3, 0.0, 0.6) * self.xlen.T, axis=0) / T.sum(self.xlen, axis=1)
        #self.att = T.sum(self.att * self.xlen.T, axis=0) / T.sum(self.xlen, axis=1)
        #self.att = theano.printing.Print("att")(self.att)
        #self.att = self.att.reshape((self.att.shape[0], self.att.shape[2]))

        # calculate the perplexity
        ff_small = T.constant(1e-20, dtype=theano.config.floatX)
        ppl_idx = softmax_out.shape[1] * \
            T.arange(softmax_out.shape[0]) + T.flatten(y_adj)
        hsum = -T.log2(T.flatten(softmax_out)[ppl_idx] + ff_small)
        hsum_new = hsum.reshape((s.shape[0], s.shape[1])).T
        self.perplexity_sentence = 2 ** (T.sum(hsum_new,
                                               axis=1) / T.sum(self.xlen, axis=1))
        self.perplexity_batch = 2 ** (T.sum(hsum *
                                            T.flatten(self.xlen.T)) / T.sum(self.xlen))
        self.perplexity_batch_v = T.sum(hsum * T.flatten(self.xlen.T))
        self.perplexity_batch_n = T.sum(self.xlen)

        # build the single step code
        h_hid = T.matrix("h_hid")
        h_cell = T.matrix("h_cell")
        x_drop_val = T.ones(
            (self.conf['emb_size'],), dtype=theano.config.floatX)
        y_drop_val = T.ones(
            (self.conf['lstm_hidden_size'],), dtype=theano.config.floatX)
        use_v = T.iscalar("use_v")
        word_t_s = T.iscalar("word_t_s")
        senti_os = T.vector("senti_os")
        one_step_theano = recurrance(word_t_s, x_drop_val, y_drop_val, use_v, T.zeros_like(
            word_t_s), h_hid, h_cell, v_i, senti_os)
        self.one_step = theano.function(
            [word_t_s, use_v, h_hid, h_cell, v_i, senti_os], outputs=one_step_theano)

    def build_loss_function(self, distributions, y_adj):
        new_s = distributions

        if self.conf['JOINED_LOSS_FUNCTION']:
            # note: we need to use the re-ordered output of the lm (ie self.lm_new_s) this accounts for dictionary differences
            #output_vocab_len = new_s.shape[1] / 2
            #weighted_results = self.new_s[:, :output_vocab_len] * self.mm_rnn.new_s + (1.0 - self.new_s[:, :output_vocab_len]) * self.lm_new_s
            #self.new_s = T.ones_like(self.new_s) * 0.90
            sm_res = self.new_s * self.mm_rnn.new_s + \
                (1.0 - self.new_s) * self.lm_new_s
            #sm_res = self.mm_rnn.new_s
            sm_res = sm_res / T.sum(sm_res, axis=1, keepdims=True)
            #sm_res = T.nnet.softmax(weighted_results)
            #sm_res = self.lm_new_s
            loss_vec = T.nnet.categorical_crossentropy(
                sm_res, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            self.loss = T.sum(loss_vec)
            return sm_res
        else:
            # calculate the total loss for this minibatch
            #self.senti = theano.printing.Print("self.senti", ["shape"])(self.senti)
            #self.xlen = theano.printing.Print("xlen", ["shape"])(self.xlen)
            sw = T.flatten(self.is_switched_seq.T)
            #sw = T.ones_like(sw)
            # * T.flatten((T.ones_like(self.xlen.T) * self.senti) )
            loss_vec = T.nnet.categorical_crossentropy(
                new_s, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            loss_vec_2 = T.nnet.categorical_crossentropy(
                new_s, T.flatten(y_adj)) * T.flatten(self.xlen.T) * (1.0 - sw)
            #self.att = theano.printing.Print('att', ["sum"])(self.att)
            #self.att2 = theano.printing.Print('att2', ["sum"])(self.att2)
            #self.att = theano.printing.Print('att', ["shape"])(self.att)
            att_flat = T.flatten(self.att)
            att_flat2 = T.flatten(self.att2)
            if self.conf['DOMAIN_ADAPT'] == DA_SUM:
                self.loss = T.sum(loss_vec) + self.conf['LAMBDA_N'] * T.sum(loss_vec_2) + \
                    T.sum((1.0 + self.conf['LAMBDA_N']) *
                          (self.conf['LAMBDA_GAM'] * (sw * -att_flat + (1 - sw)*(-att_flat2))) * T.flatten(self.xlen.T))
            elif self.conf['DOMAIN_ADAPT'] == DA_FIXED_ALPHA:
                self.loss = T.sum(loss_vec)
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM:
                self.loss = T.sum(loss_vec)
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_2:
                self.loss = T.sum(loss_vec) + \
                    self.conf['LAMBDA_N'] * T.sum(loss_vec_2)
            elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_3:
                self.loss = T.sum(loss_vec) + self.conf['LAMBDA_N'] * T.sum(loss_vec_2) + \
                    T.sum((1.0 + self.conf['LAMBDA_N']) *
                          (self.conf['LAMBDA_GAM'] * (sw * -att_flat + (1 - sw)*(-att_flat2))) * T.flatten(self.xlen.T))

            return new_s

    # compile the functions needed to train the model
    def build_model_trainer(self):
        self.X_sh_train_mask = theano.shared(
            name="X_sh_train_mask", value=self.X_train_mask, borrow=True)
        self.X_sh_train = theano.shared(
            name="X_sh_train", value=self.X_train, borrow=True)
        self.SW_sh_train = theano.shared(
            name="SW_sh_train", value=self.SW_train, borrow=True)
        self.senti_sh_train = theano.shared(
            name="senti_sh_train", value=self.senti_train, borrow=True)
        self.V_sh_train = theano.shared(
            name="V_sh_train", value=self.V_train, borrow=True)
        self.X_sh_train_drop = theano.shared(
            name="X_sh_train_drop", value=self.X_train_drop, borrow=True)
        self.Y_sh_train_drop = theano.shared(
            name="Y_sh_train_drop", value=self.Y_train_drop, borrow=True)
        if self.conf['JOINED_MODEL']:
            self.X_sh_train_lm = theano.shared(
                name="X_sh_train_lm", value=self.X_train_lm, borrow=True)
        print(self.SW_sh_train.get_value())
        print(self.SW_sh_train.get_value().shape)

        params_train = [getattr(self, p)
                        for p in self.conf['param_names_trainable']]

        # build the list of masks (which select which rows may be backpropagated)
        params_bp_mask = []
        for name in self.conf['param_names_trainable']:
            if name in self.conf['params_bp_mask']:
                params_bp_mask.append(self.conf['params_bp_mask'][name])
            else:
                params_bp_mask.append(None)

        # storage for historical gradients
        # if not self.loaded_model or (not hasattr(self,'hist_grad') and not hasattr(self,'delta_grad')):
        self.hist_grad = [theano.shared(value=np.zeros_like(
            var.get_value()), borrow=True) for var in params_train]
        self.delta_grad = [theano.shared(value=np.zeros_like(
            var.get_value()), borrow=True) for var in params_train]

        # calculate the cost for this minibatch (add L2 reg to loss function)
        regc = T.constant(self.conf['L2_REG_CONST'],
                          dtype=theano.config.floatX)
        regatt = T.constant(
            self.conf['ATT_REG_CONST'], dtype=theano.config.floatX)
        #self.cost = self.loss + regc * np.sum(map(lambda xx: (xx ** 2).sum(), params_train)) + regatt*(T.mean(self.att))
        if self.conf['DOMAIN_ADAPT'] == DA_SUM:
            self.cost = self.loss + regc * \
                np.sum([(xx ** 2).sum() for xx in params_train])
        elif self.conf['DOMAIN_ADAPT'] == DA_FIXED_ALPHA:
            self.cost = self.loss + regc * \
                np.sum([(xx ** 2).sum() for xx in params_train])
        elif self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM or self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_2 or self.conf['DOMAIN_ADAPT'] == DA_SIMILAR_PARAM_3:
            def l2(x): return (x ** 2).sum()
            reg_term = self.conf['SIMILAR_PARAM_REG'] * (l2(self.w - self.w_sw) + l2(self.b - self.b_sw) + l2(
                self.w_lstm - self.w_lstm_sw) + l2(self.wvm_sw - self.wvm) + l2(self.bmv_sw - self.bmv))
            loss_term = self.loss + regc * \
                np.sum([(xx ** 2).sum() for xx in params_train])
            reg_term = theano.printing.Print("reg_term")(reg_term)
            loss_term = theano.printing.Print("loss_term")(loss_term)
            self.cost = loss_term + reg_term
            #self.att = theano.printing.Print("att", ["shape"])(self.att)

        # build the SGD weight updates
        batch_size_f = T.constant(
            self.conf['batch_size_val'], dtype=theano.config.floatX)
        comp_grads = T.grad(self.cost, params_train)
        comp_grads = [g/batch_size_f for g in comp_grads]
        comp_grads = [T.clip(g, -self.conf['GRAD_CLIP_SIZE'],
                             self.conf['GRAD_CLIP_SIZE']) for g in comp_grads]
        comp_grads = [self.neg_sgd * g*m if m is not None else g for g,
                      m in zip(comp_grads, params_bp_mask)]
        weight_updates = get_sgd_weight_updates(self.conf['GRAD_METHOD'], comp_grads, params_train, self.hist_grad, self.delta_grad,
                                                decay=self.conf['DECAY_RATE'], learning_rate=self.conf['LEARNING_RATE'], rho=self.conf['RHO_ADADELTA'])
        print(weight_updates)

        indx = T.iscalar("indx")
        indx_wrap = indx % (
            self.X_sh_train_drop.shape[0] - self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            self.train = theano.function([indx],
                                         outputs=[self.loss, self.cost,
                                                  self.perplexity_batch],
                                         updates=weight_updates,
                                         givens={
                self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                self.mm_rnn.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                self.mm_rnn.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                self.mm_rnn.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                self.mm_rnn.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                self.mm_rnn.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                self.lm_rnn.x: self.X_sh_train_lm[indx:indx+self.conf['batch_size_val']],
                # self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                self.lm_rnn.v: np.ones((self.conf['batch_size_val'], 1), dtype=theano.config.floatX),
                self.lm_rnn.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                self.lm_rnn.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                self.lm_rnn.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap + \
                                                         self.conf['batch_size_val']]
            },
                on_unused_input='ignore')

        else:
            if self.conf['SEMI_FORCED'] < 1:
                inputs = [indx, self.forced_word, self.neg_sgd]
            else:
                inputs = [indx]
            self.train = theano.function(inputs,
                                         outputs=[self.loss, self.cost,
                                                  self.perplexity_batch],
                                         updates=weight_updates,
                                         givens={
                                             self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                                             self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                                             self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                                             self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                                             self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                                             self.is_switched_seq: self.SW_sh_train[indx:indx+self.conf['batch_size_val']],
                                             self.senti: self.senti_sh_train[indx:indx+self.conf['batch_size_val']],
                                             self.neg_sgd: 1.0},
                                         on_unused_input='ignore')
            # self.train_neg = theano.function(inputs,
            #outputs=[self.loss, self.cost, self.perplexity_batch],
            # updates=weight_updates,
            # givens={
            # self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
            # self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
            # self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
            # self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
            # self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
            # self.is_switched_seq: self.SW_sh_train[indx:indx+self.conf['batch_size_val']],
            # self.senti: 1.0 - self.senti_sh_train[indx:indx+self.conf['batch_size_val']],
            # self.neg_sgd: -1.0},
            # on_unused_input='ignore')

        return self.train

    def build_perplexity_calculator(self):
        mX, mY = self.make_drop_masks_identity(self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            v_in = T.matrix("v_in")
            givens = {self.v: v_in,
                      self.x_drop: mX[:self.x.shape[0]],
                      self.y_drop: mY[:self.x.shape[0]],
                      self.lm_rnn.x_drop: mX[:self.x.shape[0]],
                      self.lm_rnn.y_drop: mY[:self.x.shape[0]],
                      self.mm_rnn.x_drop: mX[:self.x.shape[0]],
                      self.mm_rnn.y_drop: mY[:self.x.shape[0]],
                      self.mm_rnn.x: self.x,
                      self.mm_rnn.v: v_in,
                      self.mm_rnn.xlen: self.xlen,
                      self.lm_rnn.v: np.ones((self.conf['batch_size_val'], 1), dtype=theano.config.floatX),
                      self.lm_rnn.xlen: self.xlen,
                      self.forced_word: T.ones((self.x.shape[1]+1, self.x.shape[0]), dtype=np.int32)}
            self.get_ppl_val = theano.function(
                [self.x, v_in, self.xlen, self.lm_rnn.x],
                outputs=[self.perplexity_batch_v, self.perplexity_batch_n],
                givens=givens,
                on_unused_input='ignore')
            self.get_ppl_sent_val = theano.function(
                [self.x, v_in, self.xlen, self.lm_rnn.x],
                outputs=[self.perplexity_sentence],
                givens=givens,
                on_unused_input='ignore')
        else:
            givens = {self.x_drop: mX[:self.x.shape[0]],
                      self.y_drop: mY[:self.x.shape[0]],
                      self.forced_word: T.ones((self.x.shape[1]+1, self.x.shape[0]), dtype=np.int32)}
            self.get_ppl_val = theano.function(
                [self.x, self.v, self.xlen, self.is_switched_seq, self.senti],
                outputs=[self.perplexity_batch_v, self.perplexity_batch_n],
                givens=givens,
                on_unused_input='ignore')
            self.get_ppl_sent_val = theano.function(
                [self.x, self.v, self.xlen, self.is_switched_seq, self.senti],
                outputs=[self.perplexity_sentence],
                givens=givens,
                on_unused_input='ignore')

    def build_sentence_generator(self):
        self.gen_sentence = theano.function([self.v_single, self.nstep, self.senti], outputs=[
                                            self.wout_fb, self.att_gen], on_unused_input='ignore')

    def train_complete(self, epoch_callback=None, iter_callback=None, iter_cb_freq=0, epoch_args=None, epoch_at_iter=False):
        num_iter = 0
        epoch_number = 0
        batch_size_val = self.conf['batch_size_val']
        num_train_examples = self.num_train_examples
        while True:
            cur_idx = num_iter * batch_size_val
            cur_idx = cur_idx % (
                (num_train_examples/batch_size_val)*batch_size_val)

            # we have done a full epoch
            if cur_idx == 0 and num_iter != 0:
                epoch_number += 1

                if epoch_callback is not None:
                    res = epoch_callback(self, epoch_number, epoch_args)
                    if res is not None:
                        return res

                # randomize the dataset
                idx = np.arange(num_train_examples)
                np.random.shuffle(idx)
                self.X_sh_train.set_value(
                    self.X_sh_train.get_value(borrow=True)[idx], borrow=True)
                self.V_sh_train.set_value(
                    self.V_sh_train.get_value(borrow=True)[idx], borrow=True)
                self.X_sh_train_mask.set_value(
                    self.X_sh_train_mask.get_value(borrow=True)[idx], borrow=True)
                self.SW_sh_train.set_value(
                    self.SW_sh_train.get_value(borrow=True)[idx], borrow=True)
                self.senti_sh_train.set_value(
                    self.senti_sh_train.get_value(borrow=True)[idx], borrow=True)
                if self.conf['JOINED_MODEL']:
                    self.X_sh_train_lm.set_value(
                        self.X_sh_train_lm.get_value(borrow=True)[idx], borrow=True)

                self.make_new_train_drop_masks()
            if self.conf['SEMI_FORCED'] < 1:
                sf = np.array(np.random.binomial(1, self.conf['SEMI_FORCED'], size=(
                    self.conf['MAX_SENTENCE_LEN']+1, batch_size_val)), dtype=np.int32)
                tr = self.train(cur_idx, sf)
            else:
                tr = self.train(cur_idx)
                #tr2 = self.train_neg(cur_idx)
            print(tr, num_iter * batch_size_val / float(num_train_examples))
            # print tr2, num_iter * batch_size_val / float(num_train_examples)

            if iter_cb_freq != 0 and iter_callback is not None and num_iter % iter_cb_freq == 0:
                iter_callback(self, num_iter * batch_size_val /
                              float(num_train_examples))
                if epoch_at_iter:
                    res = epoch_callback(self, epoch_number, epoch_args)
                    if res is not None:
                        return res

            num_iter += 1

    def get_val_perplexity(self, base=False):
        batch_size_val = self.conf['batch_size_val']
        num_batches = self.X_valid.shape[0] / batch_size_val
        if self.X_valid.shape[0] % batch_size_val != 0:
            num_batches += 1
        ppl_v_total = 0.0
        ppl_n_total = 0.0
        for i in range(num_batches):
            ii = i * batch_size_val
            if self.conf['JOINED_MODEL']:
                cv_X = self.convert_X_to_lm_vocab(
                    self.X_valid[ii:ii+batch_size_val])
                ppl_v, ppl_n = self.get_ppl_val(self.X_valid[ii:ii+batch_size_val],
                                                self.V_valid[ii:ii +
                                                             batch_size_val],
                                                self.X_valid_mask[ii:ii+batch_size_val], cv_X)
            else:
                senti_in = self.senti_valid[ii:ii+batch_size_val]
                if base == True:
                    senti_in = np.ones_like(senti_in) * -1.0
                sw_in = np.zeros_like(self.SW_valid[ii:ii+batch_size_val])
                ppl_v, ppl_n = self.get_ppl_val(self.X_valid[ii:ii+batch_size_val],
                                                self.V_valid[ii:ii +
                                                             batch_size_val],
                                                self.X_valid_mask[ii:ii +
                                                                  batch_size_val],
                                                sw_in,
                                                senti_in)
            ppl_v_total += ppl_v
            ppl_n_total += ppl_n
        return 2 ** (ppl_v_total / ppl_n_total)

    def get_sentence_perplexity(self, sen, v=None):
        self.setup_dataprovider(load_vocab=False)
        if v == None:
            v = np.zeros((self.conf['visual_size'],),
                         dtype=theano.config.floatX)

        x_pad, x_len = self.dp.make_single_data_instance(
            sen, self.conf['MAX_SENTENCE_LEN'])
        x_len_mask = build_len_mask(x_len, self.conf['MAX_SENTENCE_LEN']+1)

        x_pad = np.array([x_pad], dtype=np.int32)
        v = np.array([v], dtype=theano.config.floatX)
        x_len_mask = np.array([x_len_mask], dtype=theano.config.floatX)

        ppl_v, ppl_n = self.get_ppl_val(x_pad, v, x_len_mask)
        return 2 ** (ppl_v / ppl_n)

    def get_sentence_perplexity_batch(self, sens, v=None):
        self.setup_dataprovider(load_vocab=False)
        if v == None:
            v = np.zeros(
                (len(sens), self.conf['visual_size']), dtype=theano.config.floatX)

        x_pad = []
        x_len_mask = []
        for sen in sens:
            x_pad_t, x_len_t = self.dp.make_single_data_instance(
                sen, self.conf['MAX_SENTENCE_LEN'])
            x_len_mask_t = build_len_mask(
                x_len_t, self.conf['MAX_SENTENCE_LEN']+1)
            x_pad.append(x_pad_t)
            x_len_mask.append(x_len_mask_t)

        x_pad = np.array(x_pad, dtype=np.int32)
        v = np.array(v, dtype=theano.config.floatX)
        x_len_mask = np.array(x_len_mask, dtype=theano.config.floatX)

        ppl = self.get_ppl_sent_val(x_pad, v, x_len_mask)
        return ppl

    def do_one_step(self, v_i, last_step=None, senti=1.0):
        if last_step is not None:
            step = last_step
        else:
            step = {'word_t': 0,
                    'h_hid': np.zeros((2, self.conf['lstm_hidden_size']), dtype=theano.config.floatX),
                    'h_cell': np.zeros((2, self.conf['lstm_hidden_size']), dtype=theano.config.floatX),
                    'use_v': np.array(1, dtype=np.int32),
                    'senti': np.array([senti], dtype=theano.config.floatX)}

        hh, cc, s_t, att_out, att_out_log = self.one_step(
            step['word_t'], step['use_v'], step['h_hid'], step['h_cell'], v_i, step['senti'])

        step['h_hid'] = hh
        step['h_cell'] = cc
        step['s_t'] = s_t
        step['word_t'] = np.argmax(s_t)
        step['use_v'] = np.array(0, dtype=np.int32)
        step['att_out'] = np.exp(att_out)
        return step

    # generate a sentence by sampling from the predicted distributions
    def sample_sentence(self, v):
        sentence = []
        last_step = None
        for i in range(self.conf['MAX_SENTENCE_LEN'] + 1):
            last_step = self.do_one_step(v, last_step)
            c = np.arange(last_step['s_t'][0].shape[0])
            p = np.array(last_step['s_t'][0], dtype=np.float64)
            p = p / p.sum()
            w_i = np.random.choice(c, p=p)
            if w_i == 0:
                break
            sentence.append(self.model['i2w'][w_i])
        return sentence

    def sentence_idx_to_str(self, sen):
        sentence = []
        i2w = self.model['i2w']
        for i in sen:
            if i == 0:
                break
            sentence.append(i2w[i])
        return sentence

    def get_sentence(self, v, senti=np.array([1.0], dtype=theano.config.floatX)):
        res, att = self.gen_sentence(
            v, self.conf['MAX_SENTENCE_LEN'] + 1, senti)
        return [self.sentence_idx_to_str(res), att]


def main():
    rnn = RNNModel()
    # rnn.load_model("saved_model_test.pik")
    rnn.load_training_dataset()
    rnn.build_model_core()
    rnn.load_val_dataset()
    rnn.build_perplexity_calculator()
    # print rnn.get_val_perplexity()
    rnn.build_sentence_generator()

    rnn.build_model_trainer()

    def iter_callback(rnn, epoch):
        print("Epoch: %f" % epoch)
        for i in range(10):
            # np.zeros((1,)))
            print(rnn.get_sentence(
                rnn.V_valid[np.random.randint(rnn.V_valid.shape[0])]))

    def epoch_callback(rnn, num_epoch):
        rnn.save_model("saved_model_test_%d.pik" % num_epoch)
        rnn.get_val_perplexity()

    rnn.train_complete(
        iter_cb_freq=100, iter_callback=iter_callback, epoch_callback=epoch_callback)


if __name__ == "__main__":
    main()

# sys.exit(0)

# load the language model if needed
#USE_LM = False
# if USE_LM:
#    lm = LSTM_LM()
#    lm.init_dataset("../LangModel/flk30_not8k_sentences.pik")
#    lm.init_model()
#    lm.load_saved_params("../LangModel/saved_params_rnnlm_ep24_first_run.pik")

#scaler = StandardScaler()
#feats = scaler.fit_transform(feats)

#h_hid = T.vector("h_hid")
#h_cell = T.vector("h_cell")
