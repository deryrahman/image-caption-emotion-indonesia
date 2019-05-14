import numpy as np
import sys
import time
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import sys
import cPickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer
from nltk.corpus import brown
from nltk.align.bleu import BLEU
from mrnn_solver import *
# import matplotlib.pyplot as plt
import string
import scipy.io
import json
import mkl
from mrnn_util import *
from mrnn_io import *
from mrnn_lm_modular import *

sys.path.append("../LangModel")
sys.setrecursionlimit(50000)
mkl.set_num_threads(6)


class RNNModel:

    def __init__(self, conf_new=None):
        np.random.seed()

        conf = {}
        model = {}
        self.loaded_model = False

        #which gradient decent method to use
        conf['GRAD_METHOD'] = RMSPROP

        #config for each of the gradient decent methods
        conf['DECAY_RATE'] = 0.99
        conf['LEARNING_RATE'] = 0.001
        conf['RHO_ADADELTA'] = 0.95

        #what regularisation to apply
        conf['L2_REG_CONST'] = 1e-8
        conf['DROP_INPUT_FRACTION'] = 0.5
        conf['DROP_OUTPUT_FRACTION'] = 0.5
        conf['DROP_INPUT'] = True
        conf['DROP_OUTPUT'] = True

        #how much gradient clipping to apply
        conf['GRAD_CLIP_SIZE'] = 5

        #maximum length of a sentence
        #Note: actually this +1 because of start/end token
        conf['MAX_SENTENCE_LEN'] = 20

        conf['MIN_WORD_FREQ'] = 5

        #number of sentences per mini-batch
        conf['batch_size_val'] = 200

        conf['DATASET'] = RNNDataProvider.FLK8

        #set layer sizes
        conf['style_len'] = 128
        conf['vis_len'] = 128
        conf['num_styles'] = 10
        conf['emb_size'] = 256
        conf['lstm_hidden_size'] = 256
        conf['visual_size'] = 4096

        conf['JOINED_MODEL'] = False
        conf['SOFTMAX_OUT'] = True
        conf['JOINED_LOSS_FUNCTION'] = False

        conf['INIT_OUTPUT_BIAS'] = True

        conf['TRAIN_SPLIT'] = RNNDataProvider.TRAIN
        conf['VAL_SPLIT'] = RNNDataProvider.VAL

        conf['style_to_image_loss_factor'] = 1
        conf['MSQ_LOSS'] = False
        conf['BATCH_IMAGE_LOSS'] = True

        self.model = model
        if conf_new is not None:
            for k,v in conf_new.items():
                conf[k] = v

        self.conf = conf

    def set_as_joined_model(self, mm_rnn, lm_rnn):
        self.conf['JOINED_MODEL'] = True
        self.conf['SOFTMAX_OUT'] = False
        self.conf['JOINED_LOSS_FUNCTION'] = True

        self.mm_rnn = mm_rnn
        self.lm_rnn = lm_rnn

        #build conversion from lm dictionary to mm dictionary
        _, mm2lm, mm2lm_mask,_ = get_dictionary_a_to_b(mm_rnn.model['w2i'], mm_rnn.model['i2w'],
                                                       lm_rnn.model['w2i'], lm_rnn.model['i2w'])

        self.mm2lm = mm2lm

        #convert the output of the lm to match the order of the mm (zero out when doesn't exist)
        m2l = theano.shared(mm2lm, name="mm2lm", borrow=True)
        m2l_m = theano.shared(mm2lm_mask, name="mm2lm_mask", borrow=True)
        self.lm_new_s = lm_rnn.new_s[:, m2l] * m2l_m
        self.lm_new_s = self.lm_new_s / self.lm_new_s.sum(axis=1, keepdims=True)

    def save_model(self, filename):
        #get the model parameters and save them
        params_saved = dict([(p, getattr(self,p).get_value(borrow=True)) for p in self.param_names_saveable])
        self.model['params_saved'] = params_saved
        self.model['hist_grad'] = [p.get_value(borrow=True) for p in self.hist_grad]
        self.model['delta_grad'] = [p.get_value(borrow=True) for p in self.delta_grad]

        cPickle.dump((self.conf, self.model), open(filename, "wb"), protocol=2)

    def build_shared_layers(self, layers):
        for name, data in layers.items():
            setattr(self, name, theano.shared(name=name, value=data, borrow=True))

    def load_model(self, filename):
        self.__init__()
        self.loaded_model = True

        conf_new, self.model = cPickle.load(open(filename, "rb"))
        for k,v in conf_new.items():
            self.conf[k] = v

        #NOTE: backwards compatability only
        if 'output_size' not in self.conf:
            self.conf['output_size'] = self.conf['vocab_size']

        shared_layers = self.model['params_saved']
        self.build_shared_layers(shared_layers)
        self.hist_grad = []
        self.delta_grad = []
        for i,p in enumerate(self.model['hist_grad']):
            self.hist_grad.append(theano.shared(name="hist_grad[%d]" % i, value=p, borrow=True))
        for i,p in enumerate(self.model['delta_grad']):
            self.delta_grad.append(theano.shared(name="delta_grad[%d]" % i, value=p, borrow=True))

    #dropout masks that do nothing
    def make_drop_masks_identity(self, n_instances):
        mX = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN'] + 1, self.conf['emb_size']), dtype=theano.config.floatX)
        mY = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN'] + 1, self.conf['lstm_hidden_size']),
                    dtype=theano.config.floatX)
        return mX, mY

    def make_new_train_drop_masks(self):
        new_X_drop = make_drop_mask(self.X_train.shape[0] / 16 + 16, self.conf['MAX_SENTENCE_LEN'] + 1,
                                    self.conf['emb_size'], self.conf['DROP_INPUT_FRACTION'])
        new_Y_drop = make_drop_mask(self.X_train.shape[0] / 16 + 16, self.conf['MAX_SENTENCE_LEN'] + 1,
                                    self.conf['lstm_hidden_size'], self.conf['DROP_OUTPUT_FRACTION'])
        if hasattr(self, 'X_train_drop'):
            self.X_sh_train_drop.set_value(new_X_drop)
        else:
            self.X_train_drop = new_X_drop

        if hasattr(self, 'Y_train_drop'):
            self.Y_sh_train_drop.set_value(new_Y_drop)
        else:
            self.Y_train_drop = new_Y_drop

    def setup_dataprovider(self, load_vocab=True):
        if not hasattr(self, 'dp'):
            self.dp = RNNDataProvider(self.conf['DATASET'])
        if not self.dp.loaded:
            #read the sentences
            self.dp.read_dataset()
            #read the contexts
            self.dp.read_context()
            if not load_vocab:
                self.dp.w2i = self.model['w2i']
                self.dp.i2w = self.model['i2w']
            else:
                self.dp.build_vocab(self.conf['MIN_WORD_FREQ'])
                self.model['i2w'] = self.dp.i2w
                self.model['w2i'] = self.dp.w2i

    #change the vocabulary used to define the sentences in X
    def convert_X_to_lm_vocab(self, X):
        X_lm = np.zeros_like(X)
        last_tok = 1
        for r in xrange(X_lm.shape[0]):
            for c in xrange(X_lm.shape[1]):
                if X[r, c] in self.mm2lm:
                    X_lm[r, c] = self.mm2lm[X[r, c]]
                    if X_lm[r, c] != 0:
                        last_tok = X_lm[r,c]
                else:
                    X_lm[r, c] = last_tok
        return X_lm

    def load_training_dataset(self):
        #load the dataset into memory
        #also construct a new vocabulary (saved in self.model['w2i'])
        self.setup_dataprovider(load_vocab=True)

        #get the training dataset split
        self.X_train, self.Xlen_train, self.V_train, _ = self.dp.get_data_split(data_split=self.conf['TRAIN_SPLIT'],
                                                                                randomize=True,
                                                                                pad_len=self.conf['MAX_SENTENCE_LEN'])
        self.X_train_mask = build_len_masks(self.Xlen_train + 1, self.conf['MAX_SENTENCE_LEN'] + 1)
        self.num_train_examples = self.X_train.shape[0]

        if self.conf['JOINED_MODEL']:
            #map from the mm dictionary into the lm dictionary
            #if no dictionary entry repeat the last one (dont ever repeat START or STOP tokens)
            self.X_train_lm = self.convert_X_to_lm_vocab(self.X_train)

        #build the dropout masks
        self.make_new_train_drop_masks()

        self.conf['vocab_size'] = len(self.model['i2w'])
        self.conf['output_size'] = len(self.model['i2w'])

    def load_val_dataset(self):
        self.setup_dataprovider(load_vocab=False)
        self.X_valid, self.Xlen_valid, self.V_valid, _ = self.dp.get_data_split(data_split=self.conf['VAL_SPLIT'],
                                                                                randomize=False,
                                                                                pad_len=self.conf['MAX_SENTENCE_LEN'])
        self.X_valid_mask = build_len_masks(self.Xlen_valid + 1, self.conf['MAX_SENTENCE_LEN'] + 1)

    def build_model_core(self):

        #gradient clipping function
        self.clipg = lambda x: grad_clip(x, -self.conf['GRAD_CLIP_SIZE'], self.conf['GRAD_CLIP_SIZE'])

        if not self.loaded_model:
            shared_layers = {}

            #set the default network weights
            if not hasattr(self, 'style'):
                style_val = init_layer_k(self.conf['num_styles'], self.conf['style_len'])
                shared_layers['style'] = style_val

            if not hasattr(self, 'wstyle'):
                wstyle_val = init_layer_k(self.conf['style_len'], self.conf['emb_size'])
                shared_layers['wstyle'] = wstyle_val

            if not hasattr(self, 'wsty_to_img'):
                wsty_to_img_val = init_layer_k(self.conf['style_len'], self.conf['vis_len'])
                shared_layers['wsty_to_img'] = wsty_to_img_val

            if not hasattr(self, 'bsty_to_img'):
                bsty_to_img_val = np.zeros((self.conf['vis_len'],), dtype=theano.config.floatX)
                shared_layers['bsty_to_img'] = bsty_to_img_val

            if not hasattr(self, 'wv_to_sty'):
                wv_to_sty_val = init_layer_k(self.conf['visual_size'], self.conf['num_styles'])
                shared_layers['wv_to_sty'] = wv_to_sty_val

            if not hasattr(self, 'bv_to_sty'):
                bv_to_sty_val = np.zeros((self.conf['num_styles'], ), dtype=theano.config.floatX)
                shared_layers['bv_to_sty'] = bv_to_sty_val

            if not hasattr(self, 'wemb'):
                wemb_val = init_layer_k(self.conf['vocab_size'], self.conf['emb_size'])
                shared_layers['wemb'] = wemb_val

            if not hasattr(self, 'h0_hidden'):
                h0_hidden_val = np.zeros((self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
                shared_layers['h0_hidden'] = h0_hidden_val

            if not hasattr(self, 'h0_cell'):
                h0_cell_val = np.zeros((self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
                shared_layers['h0_cell'] = h0_cell_val

            #mapping from visual space to word space
            if not hasattr(self, 'wvm'):
                wvm_val = init_layer_k(self.conf['visual_size'], self.conf['vis_len'])
                shared_layers['wvm'] = wvm_val

            if not hasattr(self, 'bmv'):
                bmv_val = np.zeros((self.conf['emb_size'],), dtype=theano.config.floatX)
                shared_layers['bmv'] = bmv_val

            #LSTM layer parameters
            if not hasattr(self, 'w_lstm'):
                w_lstm_val = init_layer_k(self.conf['lstm_hidden_size'] * 2, self.conf['lstm_hidden_size'] * 4)
                shared_layers['w_lstm'] = w_lstm_val

            #mapping from RNN hidden output to vocabulary
            if not hasattr(self, 'w'):
                w_val = init_layer_k(self.conf['lstm_hidden_size'], self.conf['output_size'])
                shared_layers['w'] = w_val

            if not hasattr(self, 'b'):
                b_val = np.zeros((self.conf['output_size'],), dtype=theano.config.floatX)
                if self.conf["INIT_OUTPUT_BIAS"]:
                    #set the bias on the last layer to be the log prob of each of the words in the vocab
                    wcount = 0
                    w2i = self.dp.w2i
                    w2c = self.dp.get_word_counts(RNNDataProvider.TRAIN)
                    for w in w2i:
                        if w in w2c:
                            wcount += w2c[w]
                    wcount += self.X_train.shape[0]
                    b_val[w2i[RNNDataProvider.STOP_TOKEN]] = np.log(self.X_train.shape[0] / float(wcount))
                    for w in w2i:
                        if w in w2c:
                            b_val[w2i[w]] = np.log(w2c[w] / float(wcount))
                    b_val -= np.max(b_val[1:])
                shared_layers['b'] = b_val

            self.build_shared_layers(shared_layers)

        #remember the important parameters for saving + SGD
        self.param_names_saveable = ["wemb", "h0_hidden", "h0_cell", "wvm", "bmv", "w", "b",
                                     "w_lstm", "style", "wstyle", "wsty_to_img", "bsty_to_img", "wv_to_sty",
                                     "bv_to_sty"]

        #input variables for training
        self.x = T.imatrix("x")
        self.v = T.matrix("v")
        self.xlen = T.matrix("xlen")

        self.style_idx = T.ivector("v")

        #input variables for generation
        self.v_single = T.vector("v")
        self.nstep = T.iscalar("nstep")

        #the dropout masks
        #drop the input
        self.x_drop = T.tensor3("x_drop")
        #drop the output
        self.y_drop = T.tensor3("y_drop")

        #hidden layer ouput
        h_tm1 = T.vector("h_tm1")
        #word indexes
        word_t = T.ivector("word_t")
        #visual information
        v_i = T.vector("v")

        #Generates the next word based on the: previous true word, hidden state & visual features
        #inputs: hiddent_layer, last_predicted word, visual features
        def recurrance(word_t, x_drop_slice, hh_drop_slice, use_v, h_tm1_hidden, h_tm1_cell, v_i, style_idx):

            if word_t.ndim == 0:
                style_idx = style_idx[0]

            #get the word embedding matrix or the context information
            x_t = ifelse(T.eq(use_v, 1),T.concatenate([T.dot(v_i, self.wvm).T, self.style[style_idx].T]).T + self.bmv,
                         self.wemb[word_t])

            #if we are not doing minibatch training
            if word_t.ndim == 0:
                x_t = x_t.reshape((1, x_t.shape[0]))
                h_tm1_hidden = h_tm1_hidden.reshape((1, h_tm1_hidden.shape[0]))

            #dropout on the input embddings
            if self.conf['DROP_INPUT']:
                x_t *= x_drop_slice

            #clip the gradients so they dont get too large
            h_tm1_hidden_clip = self.clipg(h_tm1_hidden)

            in_state = T.concatenate([x_t, h_tm1_hidden_clip], axis=1)

            #calculate 8 dot products in one go
            dot_out = T.dot(in_state, self.w_lstm)

            lstm_hidden_size = self.conf['lstm_hidden_size']
            #input gate
            ig = T.nnet.sigmoid(dot_out[:, :lstm_hidden_size])
            #forget gate
            fg = T.nnet.sigmoid(dot_out[:, lstm_hidden_size:lstm_hidden_size * 2])
            #output gate
            og = T.nnet.sigmoid(dot_out[:, lstm_hidden_size * 2:lstm_hidden_size * 3])

            # cell memory
            cc = fg * h_tm1_cell + ig * T.tanh(dot_out[:, lstm_hidden_size * 3:])

            # hidden state
            hh = og * cc

            #drop the output state
            if self.conf['DROP_OUTPUT']:
                hh = hh * hh_drop_slice

            #the distribution over output words
            if self.conf['SOFTMAX_OUT']:
                s_t = T.nnet.softmax(T.dot(hh, self.w) + self.b)
            else:
                s_t = T.nnet.sigmoid(T.dot(hh, self.w) + self.b)

            #if we are not doing minibatch training
            if word_t.ndim == 0:
                hh = hh[0]
                cc = cc[0]

            return [hh, cc, s_t]

        #Generates the next word by feeding the old word as input
        #inputs: hiddent_layer, last_predicted word, visual features
        def recurrance_word_feedback(h_tm1_hidden, h_tm1_cell, word_t, use_visual, v_i, style_idx):
            x_drop_val = T.ones((self.conf['emb_size'],), dtype=theano.config.floatX)
            y_drop_val = T.ones((self.conf['lstm_hidden_size'],), dtype=theano.config.floatX)
            [hh, cc, s_t] = recurrance(word_t, x_drop_val, y_drop_val, use_visual, h_tm1_hidden, h_tm1_cell, v_i,
                                       style_idx)

            #the predicted word
            w_idx = T.cast(T.argmax(s_t, axis=1), dtype='int32')[0]

            return [hh, cc, s_t[0], w_idx, T.zeros((0,), dtype='int32')[0]]

        #build the teacher forcing loop
        use_visual_info = T.concatenate([T.ones((1,), dtype=np.int32), T.zeros((self.conf['MAX_SENTENCE_LEN'],),
                                         dtype=np.int32)])
        h0_hidden_matrix = self.h0_hidden * T.ones((self.x.shape[0], self.h0_hidden.shape[0]))
        h0_cell_matrix = self.h0_cell * T.ones((self.x.shape[0], self.h0_cell.shape[0]))
        x_adj = T.concatenate([T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype), self.x.T])
        y_adj = T.concatenate([self.x.T, T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype)])
        [_, _, s], _ = theano.scan(fn=recurrance,
                                   sequences=[x_adj, self.x_drop.dimshuffle((1, 0, 2)),
                                              self.y_drop.dimshuffle((1, 0, 2)), use_visual_info],
                                   n_steps=self.conf['MAX_SENTENCE_LEN'] + 1,
                                   non_sequences=[self.v, self.style_idx],
                                   outputs_info=[h0_hidden_matrix, h0_cell_matrix, None])

        #build the un-forced loop
        [_, _, _, self.wout_fb, _], _ = theano.scan(fn=recurrance_word_feedback,
                                                    non_sequences=[self.v_single, self.style_idx],
                                                    outputs_info=[self.h0_hidden, self.h0_cell, None,
                                                                  np.array(0, dtype=np.int32),
                                                                  T.ones((1,), dtype=np.int32)[0]],
                                                    n_steps=self.nstep)

        #TODO: try going from style to image in the batch
        #try to generate the image features from the style information

        if self.conf['MSQ_LOSS']:
            styin = grad_reverse(self.style[self.style_idx])
            v_guess = T.dot(styin, self.wsty_to_img) + self.bsty_to_img
            v_real = T.dot(self.v, grad_ignore(self.wvm))
            msq_diff = T.sum((v_guess - v_real)**2) / self.v.shape[0]
        elif self.conf['BATCH_IMAGE_LOSS']:
            sty_guess = T.dot(self.v, self.wv_to_sty) + self.bv_to_sty
            sty_guess_sm = T.nnet.softmax(sty_guess)
            self.msq_diff = T.sum(T.nnet.categorical_crossentropy(sty_guess_sm, self.style_idx))
            self.style_guess_correct = T.sum(T.eq(T.argmax(sty_guess_sm, axis=1), self.style_idx)) / T.cast(self.v.shape[0], dtype=theano.config.floatX)
        else:
            msq_diff = None

        self.new_s = s.reshape((s.shape[0] * s.shape[1], s.shape[2]))
        softmax_out = self.build_loss_function(self.new_s, y_adj, self.msq_diff)

        #calculate the perplexity
        ff_small = T.constant(1e-20, dtype=theano.config.floatX)
        ppl_idx = softmax_out.shape[1] * T.arange(softmax_out.shape[0]) + T.flatten(y_adj)
        hsum = -T.log2(T.flatten(softmax_out)[ppl_idx] + ff_small)
        self.hsum_new = hsum.reshape((s.shape[0], s.shape[1])).T * self.xlen
        self.perplexity_sentence = 2 ** (T.sum(self.hsum_new, axis=1) / T.sum(self.xlen, axis=1))
        self.perplexity_batch = 2 ** (T.sum(hsum * T.flatten(self.xlen.T)) / T.sum(self.xlen))
        self.perplexity_batch_v = T.sum(hsum * T.flatten(self.xlen.T))
        self.perplexity_batch_n = T.sum(self.xlen)

        #build the single step code
        h_hid = T.vector("h_hid")
        h_cell = T.vector("h_cell")
        x_drop_val = T.ones((self.conf['emb_size'],), dtype=theano.config.floatX)
        y_drop_val = T.ones((self.conf['lstm_hidden_size'],), dtype=theano.config.floatX)
        use_v = T.iscalar("use_v")
        word_t_s = T.iscalar("word_t_s")
        #one_step_theano = recurrance(word_t_s, x_drop_val, y_drop_val, use_v, h_hid, h_cell, v_i)
        #self.one_step = theano.function([word_t_s, use_v, h_hid, h_cell, v_i], outputs=one_step_theano)

    def build_loss_function(self, distributions, y_adj, msq_diff=None):
        new_s = distributions

        if self.conf['JOINED_LOSS_FUNCTION']:
            #note: we need to use the re-ordered output of the lm (ie self.lm_new_s) this accounts
            # for dictionary differences
            #output_vocab_len = new_s.shape[1] / 2
            #weighted_results = self.new_s[:, :output_vocab_len] * self.mm_rnn.new_s +
            # (1.0 - self.new_s[:, :output_vocab_len]) * self.lm_new_s
            sm_res = self.new_s * self.mm_rnn.new_s + (1.0 - self.new_s) * self.lm_new_s
            #sm_res = T.nnet.softmax(weighted_results)
            #sm_res = self.lm_new_s
            loss_vec = T.nnet.categorical_crossentropy(sm_res, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            self.loss = T.sum(loss_vec)
            return sm_res
        else:
            #calculate the total loss for this minibatch
            loss_vec = T.nnet.categorical_crossentropy(new_s, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            self.loss = T.sum(loss_vec)
            if msq_diff is not None:
                sty_fact = T.constant(self.conf['style_to_image_loss_factor'])
                self.loss += msq_diff * sty_fact

            return new_s

    #compile the functions needed to train the model
    def build_model_trainer(self):
        self.X_sh_train_mask = theano.shared(name="X_sh_train_mask", value=self.X_train_mask, borrow=True)
        self.X_sh_train = theano.shared(name="X_sh_train",value=self.X_train, borrow=True)
        self.V_sh_train = theano.shared(name="V_sh_train",value=self.V_train, borrow=True)
        self.X_sh_train_drop = theano.shared(name="X_sh_train_drop",value=self.X_train_drop, borrow=True)
        self.Y_sh_train_drop = theano.shared(name="Y_sh_train_drop",value=self.Y_train_drop, borrow=True)
        if self.conf['JOINED_MODEL']:
            self.X_sh_train_lm = theano.shared(name="X_sh_train_lm",value=self.X_train_lm, borrow=True)


        #set the parameters we want to train
        self.param_names_trainable = ["wemb", "wvm", "bmv", "w", "b", "w_lstm", "style", "wstyle", "wsty_to_img", "bsty_to_img", "wv_to_sty", "bv_to_sty"]
        params_train = [getattr(self, p) for p in self.param_names_trainable]

        #storage for historical gradients
        self.hist_grad = [theano.shared(value=np.zeros_like(var.get_value()), borrow=True) for var in params_train]
        self.delta_grad = [theano.shared(value=np.zeros_like(var.get_value()), borrow=True) for var in params_train]

        #calculate the cost for this minibatch (add L2 reg to loss function)
        regc = T.constant(self.conf['L2_REG_CONST'], dtype=theano.config.floatX)
        self.cost = self.loss + regc * np.sum(map(lambda xx: (xx ** 2).sum(), params_train))

        #build the SGD weight updates
        batch_size_f = T.constant(self.conf['batch_size_val'], dtype=theano.config.floatX)
        comp_grads = T.grad(self.cost, params_train)
        comp_grads = [g/batch_size_f for g in comp_grads]
        comp_grads = [T.clip(g, -self.conf['GRAD_CLIP_SIZE'], self.conf['GRAD_CLIP_SIZE']) for g in comp_grads]
        weight_updates = get_sgd_weight_updates(self.conf['GRAD_METHOD'], comp_grads, params_train, self.hist_grad, self.delta_grad,
                                        decay=self.conf['DECAY_RATE'], learning_rate=self.conf['LEARNING_RATE'])

        indx = T.iscalar("indx")
        indx_wrap = indx % (self.X_sh_train_drop.shape[0] - self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            self.train = theano.function([indx],
                        outputs=[self.loss, self.cost, self.perplexity_batch],
                        updates=weight_updates,
                        givens={
                            self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.v: T.zeros_like(self.V_sh_train[indx:indx+self.conf['batch_size_val']]),
                            self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.mm_rnn.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.mm_rnn.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.mm_rnn.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.mm_rnn.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.mm_rnn.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.lm_rnn.x: self.X_sh_train_lm[indx:indx+self.conf['batch_size_val']],
                            self.lm_rnn.v: np.ones((self.conf['batch_size_val'], 1), dtype=theano.config.floatX),#self.V_sh_train[indx:indx+self.conf['batch_size_val']], 
                            self.lm_rnn.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.lm_rnn.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.lm_rnn.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']]
                            },
                        on_unused_input='ignore')

        else:
            self.test_style = theano.function([indx, self.style_idx],
                        outputs=[self.perplexity_sentence],
                        givens={
                            self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']]},
                        on_unused_input='ignore')
            self.train = theano.function([indx, self.style_idx],
                        outputs=[self.loss, self.cost, self.perplexity_batch, self.msq_diff, self.style_guess_correct],
                        updates=weight_updates,
                        givens={
                            self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                            self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']]},
                        on_unused_input='ignore')

        return self.train

    def build_perplexity_calculator(self):
        mX, mY = self.make_drop_masks_identity(self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            v_in = T.matrix("v_in")
            self.get_ppl_val = theano.function(
                                    [self.x, v_in, self.xlen, self.lm_rnn.x],
                                    outputs=[self.perplexity_batch_v, self.perplexity_batch_n],
                                    givens={self.v : T.zeros_like(v_in),
                                            self.x_drop: mX[:self.x.shape[0]],
                                            self.y_drop : mY[:self.x.shape[0]],
                                            self.lm_rnn.x_drop : mX[:self.x.shape[0]],
                                            self.lm_rnn.y_drop : mY[:self.x.shape[0]],
                                            self.mm_rnn.x_drop: mX[:self.x.shape[0]],
                                            self.mm_rnn.y_drop : mY[:self.x.shape[0]],
                                            self.mm_rnn.x : self.x,
                                            self.mm_rnn.v : v_in,
                                            self.mm_rnn.xlen : self.xlen,
                                            self.lm_rnn.v : np.ones((self.conf['batch_size_val'], 1), dtype=theano.config.floatX),
                                            self.lm_rnn.xlen : self.xlen},

                                    on_unused_input='ignore')
        else:
            self.get_ppl_val = theano.function(
                                    [self.x, self.v, self.xlen, self.style_idx],
                                    outputs=[self.perplexity_batch_v, self.perplexity_batch_n],
                                    givens={self.x_drop: mX[:self.x.shape[0]], self.y_drop : mY[:self.x.shape[0]]},
                                    on_unused_input='ignore')

    def build_sentence_generator(self):
        self.gen_sentence = theano.function([self.v_single, self.nstep, self.style_idx], outputs=self.wout_fb)

    def train_complete(self, epoch_callback = None, iter_callback=None, iter_cb_freq = 0):
        num_iter = 0
        epoch_number = 0
        batch_size_val = self.conf['batch_size_val']
        num_train_examples = self.num_train_examples
        while True:
            cur_idx = num_iter * batch_size_val
            cur_idx = cur_idx % ((num_train_examples/batch_size_val)*batch_size_val)

            #we have done a full epoch
            if cur_idx == 0 and num_iter != 0:
                epoch_number += 1

                if epoch_callback is not None:
                    epoch_callback(self, epoch_number)

                #randomize the dataset
                idx = np.arange(num_train_examples)
                np.random.shuffle(idx)
                self.X_sh_train.set_value(self.X_sh_train.get_value(borrow=True)[idx], borrow=True)
                self.V_sh_train.set_value(self.V_sh_train.get_value(borrow=True)[idx], borrow=True)
                self.X_sh_train_mask.set_value(self.X_sh_train_mask.get_value(borrow=True)[idx], borrow=True)
                if self.conf['JOINED_MODEL']:
                    self.X_sh_train_lm.set_value(self.X_sh_train_lm.get_value(borrow=True)[idx], borrow=True)

                self.make_new_train_drop_masks()

            all_style_scores = np.zeros((self.conf['num_styles'], batch_size_val), dtype=theano.config.floatX)
            for i in xrange(self.conf['num_styles']):
                style_idx = np.ones((batch_size_val,), dtype=np.int32) * i
                style_scores = self.test_style(cur_idx, style_idx)[0]
                all_style_scores[i, :] = style_scores
            best_styles = np.array(np.argmin(all_style_scores, axis=0), dtype=np.int32)
            print np.bincount(best_styles)


            res = self.train(cur_idx, best_styles)

            print res, num_iter * batch_size_val / float(num_train_examples)

            if iter_cb_freq != 0 and iter_callback is not None and num_iter % iter_cb_freq == 0:
                iter_callback(self, num_iter * batch_size_val / float(num_train_examples))

            num_iter += 1

    def get_val_perplexity(self):
        batch_size_val = self.conf['batch_size_val']
        num_batches = self.X_valid.shape[0] / batch_size_val
        if self.X_valid.shape[0] % batch_size_val != 0: num_batches+=1
        ppl_v_total = 0.0
        ppl_n_total = 0.0
        for i in xrange(num_batches):
            ii = i * batch_size_val
            if self.conf['JOINED_MODEL']:
                cv_X = self.convert_X_to_lm_vocab(self.X_valid[ii:ii+batch_size_val])
                ppl_v, ppl_n = self.get_ppl_val(self.X_valid[ii:ii+batch_size_val],
                                            self.V_valid[ii:ii+batch_size_val],
                                            self.X_valid_mask[ii:ii+batch_size_val], cv_X)
            else:
                ppl_v, ppl_n = self.get_ppl_val(self.X_valid[ii:ii+batch_size_val],
                                            self.V_valid[ii:ii+batch_size_val],
                                            self.X_valid_mask[ii:ii+batch_size_val])
            ppl_v_total += ppl_v
            ppl_n_total += ppl_n
        return 2 ** (ppl_v_total / ppl_n_total)

    def get_sentence_perplexity(self, sen, v = None):
        self.setup_dataprovider(load_vocab=False)
        if v == None:
            v = np.zeros((self.conf['visual_size'],), dtype=theano.config.floatX)

        x_pad,x_len = self.dp.make_single_data_instance(sen, self.conf['MAX_SENTENCE_LEN'])
        x_len_mask = build_len_mask(x_len, self.conf['MAX_SENTENCE_LEN']+1)

        x_pad = np.array([x_pad], dtype=np.int32)
        v = np.array([v], dtype=theano.config.floatX)
        x_len_mask = np.array([x_len_mask], dtype=theano.config.floatX)

        ppl_v, ppl_n = self.get_ppl_val(x_pad, v, x_len_mask)
        return 2 ** (ppl_v / ppl_n)

    def get_sentence_perplexity_batch(self, sens, v = None):
        #TODO: finish this function needs a batch perplexity theano function to be implemented
        self.setup_dataprovider(load_vocab=False)
        if v == None:
            v = np.zeros((len(sens), self.conf['visual_size']), dtype=theano.config.floatX)

        x_pad = []
        x_len_mask = []
        for sen in sens:
            x_pad_t,x_len_t = self.dp.make_single_data_instance(sen, self.conf['MAX_SENTENCE_LEN'])
            x_len_mask_t = build_len_mask(x_len_t, self.conf['MAX_SENTENCE_LEN']+1)
            x_pad.append(x_pad_t)
            x_len_mask.append(x_len_mask_t)

        x_pad = np.array(x_pad, dtype=np.int32)
        v = np.array(v, dtype=theano.config.floatX)
        x_len_mask = np.array(x_len_mask, dtype=theano.config.floatX)

        ppl_v, ppl_n = self.get_ppl_val(x_pad, v, x_len_mask)
        return 2 ** (ppl_v / ppl_n)

    def do_one_step(self, v_i, last_step = None):
        if last_step is not None:
            step = last_step
        else:
            step = {'word_t': 0,
              'h_hid': np.zeros(self.conf['lstm_hidden_size'], dtype=theano.config.floatX),
              'h_cell': np.zeros(self.conf['lstm_hidden_size'], dtype=theano.config.floatX),
              'use_v': np.array(1, dtype=np.int32)}

        hh, cc, s_t = self.one_step(step['word_t'], step['use_v'], step['h_hid'], step['h_cell'], v_i)

        step['h_hid'] = hh
        step['h_cell'] = cc
        step['s_t'] = s_t
        step['word_t'] = np.argmax(s_t)
        step['use_v'] = np.array(0, dtype=np.int32)
        return step


    def sentence_idx_to_str(self, sen):
        sentence = []
        i2w = self.model['i2w']
        for i in sen:
            if i == 0:break
            sentence.append(i2w[i])
        return sentence

    def get_sentence(self, v, style_idx):
        res = self.gen_sentence(v, self.conf['MAX_SENTENCE_LEN'] + 1, style_idx)
        return self.sentence_idx_to_str(res)



def main():
    rnn = RNNModel()
    #rnn.load_model("saved_model_test.pik")
    rnn.load_training_dataset()
    rnn.build_model_core()
    rnn.load_val_dataset()
    rnn.build_perplexity_calculator()
    #print rnn.get_val_perplexity()
    rnn.build_sentence_generator()

    rnn.build_model_trainer()
    def iter_callback(rnn, epoch):
        print "Epoch: %f" % epoch
        for i in xrange(10):
            print rnn.get_sentence(rnn.V_valid[np.random.randint(rnn.V_valid.shape[0])])#np.zeros((1,)))

    def epoch_callback(rnn, num_epoch):
        rnn.save_model("saved_model_test_%d.pik" % num_epoch)
        rnn.get_val_perplexity()

    rnn.train_complete(iter_cb_freq=100, iter_callback=iter_callback, epoch_callback=epoch_callback)

if __name__ == "__main__":
    main()

#sys.exit(0)


#load the language model if needed
#USE_LM = False
#if USE_LM:
#    lm = LSTM_LM()
#    lm.init_dataset("../LangModel/flk30_not8k_sentences.pik")
#    lm.init_model()
#    lm.load_saved_params("../LangModel/saved_params_rnnlm_ep24_first_run.pik")


#scaler = StandardScaler()
#feats = scaler.fit_transform(feats)

#h_hid = T.vector("h_hid")
#h_cell = T.vector("h_cell")
