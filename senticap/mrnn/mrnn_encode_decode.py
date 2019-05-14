import numpy as np

import sys
sys.setrecursionlimit(50000)

import time
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import sys
import cPickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer
from nltk.corpus import brown
from mrnn_solver import *
# import matplotlib.pyplot as plt
import string

import scipy.io
import json

import mkl
mkl.set_num_threads(6)

from mrnn_util import *
from mrnn_io import *
sys.path.append("../LangModel")
from mrnn_lm_modular import *

class RNNModel:

    ENCODE_EXTRA_NONE = "tokens"
    ENCODE_EXTRA_TITLE = "title"
    ENCODE_EXTRA_DESC = "desc"
    ENCODE_EXTRA_TAGS = "tags"

    def __init__(self, conf_new = None, encoder=None):
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
        conf['DROP_MASK_SIZE_DIV'] = 16

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
        conf['emb_size'] = 256
        conf['lstm_hidden_size'] = 256
        conf['visual_size'] = 4096

        conf['JOINED_MODEL'] = False
        conf['SOFTMAX_OUT'] = True
        conf['JOINED_LOSS_FUNCTION'] = False

        conf['INIT_OUTPUT_BIAS'] = True

        conf['TRAIN_SPLIT'] = RNNDataProvider.TRAIN
        conf['VAL_SPLIT'] = RNNDataProvider.VAL

        conf['BATCH_NORM'] = False

        conf['SEMI_FORCED'] = 1

        conf['REVERSE_TEXT'] = False

        conf['DECODER'] = False

        conf['ENCODE_EXTRA'] = self.ENCODE_EXTRA_NONE

        conf['params_bp_mask'] = {}

        #remember the important parameters for saving + SGD
        conf['param_names_saveable'] = ["wemb", "h0_hidden", "h0_cell", "wvm", "bmv", "w", "b", "w_lstm"]
        if conf['BATCH_NORM']:
           conf['param_names_saveable'].append("gamma_h")
           conf['param_names_saveable'].append("beta_h")

        #set the parameters we want to train
        conf['param_names_trainable'] = ["wemb", "wvm", "bmv", "w", "b", "w_lstm"]
        if conf['BATCH_NORM']:
            conf['param_names_trainable'].append("gamma_h")
            conf['param_names_trainable'].append("beta_h")

        self.model = model
        if conf_new is not None:
            for k,v in conf_new.items():
                conf[k] = v

        self.conf = conf

        self.encoder = encoder

    def set_as_joined_model(self, mm_rnn, lm_rnn):
        self.conf['JOINED_MODEL'] = True
        self.conf['SOFTMAX_OUT'] = False
        self.conf['JOINED_LOSS_FUNCTION'] = True

        self.mm_rnn = mm_rnn
        self.lm_rnn = lm_rnn

        #build conversion from lm dictionary to mm dictionary
        mm2lm_map, mm2lm, mm2lm_mask,_ =  get_dictionary_a_to_b(mm_rnn.model['w2i'], mm_rnn.model['i2w'],
                                                        lm_rnn.model['w2i'], lm_rnn.model['i2w'])

        self.mm2lm = mm2lm
        self.mm2lm_map = mm2lm_map

        #convert the output of the lm to match the order of the mm (zero out when doesn't exist)
        m2l = theano.shared(mm2lm, name="mm2lm", borrow=True)
        m2l_m = theano.shared(mm2lm_mask, name="mm2lm_mask", borrow=True)
        self.lm_new_s = lm_rnn.new_s[:, m2l] * m2l_m
        self.lm_new_s = self.lm_new_s / self.lm_new_s.sum(axis=1, keepdims=True)



    def save_model(self, filename, to_file=True):
        #get the model parameters and save them
        params_saved = dict([(p, getattr(self,p).get_value(borrow=True)) for p in self.conf['param_names_saveable']])
        self.model['params_saved'] = params_saved
        self.model['hist_grad'] = {}
        self.model['delta_grad'] = {}
        for p in self.conf['param_names_saveable']:
            if p in self.conf['param_names_trainable']:
                idx = self.conf['param_names_trainable'].index(p)
                self.model['hist_grad'][p] = self.hist_grad[idx].get_value(borrow=True)
                self.model['delta_grad'][p] = self.delta_grad[idx].get_value(borrow=True)

        if self.conf['DECODER']:
            encoder_data = self.encoder.save_model("", to_file=False)
            cPickle.dump((self.conf, self.model, encoder_data), open(filename, "wb"), protocol=2)
        else:
            if to_file:
                cPickle.dump((self.conf, self.model), open(filename, "wb"), protocol=2)
            else:
                return (self.conf, self.model)

    def build_shared_layers(self, layers):
        ne = ""
        if not self.conf['DECODER']:
            ne = "_encoder"
        for name, data in layers.items():
            setattr(self, name, theano.shared(name=name+ne, value=data, borrow=True))

    def load_model(self, filename, conf=None, load_solver_params=True):
        print filename
        data = cPickle.load(open(filename, "rb"))
        if len(data) == 3:
            encoder_data = data[2]
            data = (data[0], data[1])
            self.load_model_from_data(data, conf, load_solver_params)
            self.encoder = RNNModel()
            self.encoder.load_model_from_data(encoder_data, conf, load_solver_params)
        else:
            self.load_model_from_data(data, conf, load_solver_params)
        
    def load_model_from_data(self, data, conf = None, load_solver_params=True):
        self.__init__()
        self.loaded_model = True

        conf_new, self.model = data
        for k,v in conf_new.items():
            self.conf[k] = v
        if conf is not None:
            for k,v in conf.items():
                self.conf[k] = v


        shared_layers = self.model['params_saved']
        self.build_shared_layers(shared_layers)

        if load_solver_params:
            self.hist_grad = []
            self.delta_grad = []
            for i, p in enumerate(self.conf['param_names_trainable']):
                if p in self.model['hist_grad']:
                    v = self.model['hist_grad'][p]
                    self.hist_grad.append(theano.shared(name="hist_grad[%d]" % i, value=v, borrow=True))
                else:
                    v = np.zeros_like(getattr(self, p).get_value(borrow=True))
                    self.hist_grad.append(theano.shared(name="hist_grad[%d]" % i, value=v, borrow=True))
                if p in self.model['delta_grad']:
                    v = self.model['delta_grad'][p]
                    self.delta_grad.append(theano.shared(name="delta_grad[%d]" % i, value=v, borrow=True))
                else:
                    v = np.zeros_like(getattr(self, p).get_value(borrow=True))
                    self.delta_grad.append(theano.shared(name="delta_grad[%d]" % i, value=v, borrow=True))

    #dropout masks that do nothing
    def make_drop_masks_identity(self, n_instances):
        mX = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN']+1, self.conf['emb_size']), dtype=theano.config.floatX)
        mY = T.ones((n_instances, self.conf['MAX_SENTENCE_LEN']+1, self.conf['lstm_hidden_size']), dtype=theano.config.floatX)
        return mX, mY
        
    def make_new_train_drop_masks(self):
        div = self.conf['DROP_MASK_SIZE_DIV']
        num_drop_masks = self.X_train.shape[0]/div + div
        if num_drop_masks < self.conf['batch_size_val']:
            num_drop_masks = self.conf['batch_size_val']
        new_X_drop = make_drop_mask(num_drop_masks, self.conf['MAX_SENTENCE_LEN']+1, self.conf['emb_size'], self.conf['DROP_INPUT_FRACTION'])
        new_Y_drop = make_drop_mask(num_drop_masks, self.conf['MAX_SENTENCE_LEN']+1, self.conf['lstm_hidden_size'], self.conf['DROP_OUTPUT_FRACTION'])
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
            self.dp = RNNDataProvider(self.conf['DATASET'], self.conf['REVERSE_TEXT'])
        if not self.dp.loaded:
            #read the sentences
            self.dp.read_dataset()
            #read the contexts
            self.dp.read_context()
            if not load_vocab:
                self.dp.w2i = self.model['w2i']
                self.dp.i2w = self.model['i2w']
            else:
                self.dp.build_vocab(self.conf['MIN_WORD_FREQ'], source=self.conf['ENCODE_EXTRA'])

                self.model['i2w'] = self.dp.i2w
                self.model['w2i'] = self.dp.w2i


    #change the vocabulary used to define the sentences in X
    def convert_X_to_lm_vocab(self, X):
        X_lm = np.zeros_like(X)
        last_tok = 1
        for r in xrange(X_lm.shape[0]):
            for c in xrange(X_lm.shape[1]):
                if X[r, c] in self.mm2lm_map:
                    X_lm[r, c] = self.mm2lm_map[X[r, c]]
                    if X_lm[r, c] != 0:
                        last_tok = X_lm[r,c]
                else:
                    X_lm[r, c] = last_tok
            #print X_lm[r]
            #if np.random.randint(15) == 1:
            #    sys.exit(0)
        return X_lm


    def load_training_dataset(self, rotate = False):
        #load the dataset into memory
        #also construct a new vocabulary (saved in self.model['w2i'])
        self.setup_dataprovider(load_vocab=True)

        #get the training dataset split
        np.random.seed(123)
        self.X_train, self.Xlen_train, self.V_train, _ = self.dp.get_data_split(data_split=self.conf['TRAIN_SPLIT'], randomize=True, pad_len=self.conf['MAX_SENTENCE_LEN'], rotate_X_with_id = rotate, source=self.conf['ENCODE_EXTRA'])
        self.X_train_mask = build_len_masks(self.Xlen_train+1, self.conf['MAX_SENTENCE_LEN']+1)
        self.num_train_examples = self.X_train.shape[0]

        if self.conf['JOINED_MODEL']: 
            #map from the mm dictionary into the lm dictionary
            #if no dictionary entry repeat the last one (dont ever repeat START or STOP tokens)
            self.X_train_lm = self.convert_X_to_lm_vocab(self.X_train)

        #build the dropout masks
        if self.conf['DECODER']:
            self.make_new_train_drop_masks()

        self.conf['vocab_size'] = len(self.model['i2w'])
        self.conf['output_size'] = len(self.model['i2w'])
        
    def load_val_dataset(self, rotate=False):
        self.setup_dataprovider(load_vocab=False)
        self.X_valid, self.Xlen_valid, self.V_valid, self.Id_valid = self.dp.get_data_split(
                data_split=self.conf['VAL_SPLIT'], 
                randomize=False, 
                pad_len=self.conf['MAX_SENTENCE_LEN'],
                rotate_X_with_id=rotate,
                source=self.conf['ENCODE_EXTRA'])
        self.X_valid_mask = build_len_masks(self.Xlen_valid+1, self.conf['MAX_SENTENCE_LEN']+1)

    def build_model_core(self):

        #gradient clipping function
        self.clipg = lambda x: grad_clip(x, -self.conf['GRAD_CLIP_SIZE'], self.conf['GRAD_CLIP_SIZE'])

        shared_layers = {}

        if self.conf['BATCH_NORM']:
            if not hasattr(self, 'gamma_h'):
                gamma_h_val = np.ones((self.conf['lstm_hidden_size'] * 2,), dtype=theano.config.floatX)
                shared_layers['gamma_h'] = gamma_h_val
            if not hasattr(self, 'beta_h'):
                beta_h_val = np.zeros((self.conf['lstm_hidden_size'] * 2,), dtype=theano.config.floatX)
                shared_layers['beta_h'] = beta_h_val

        #set the default network weights
        if not hasattr(self, 'wemb'):
            wemb_val = init_layer_k(self.conf['vocab_size'], self.conf['emb_size'])
            shared_layers['wemb'] = wemb_val;
        
        if not hasattr(self, 'h0_hidden'):
            h0_hidden_val = np.zeros((self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
            shared_layers['h0_hidden'] = h0_hidden_val

        if not hasattr(self, 'h0_cell'):
            h0_cell_val = np.zeros((self.conf['lstm_hidden_size'], ), dtype=theano.config.floatX)
            shared_layers['h0_cell'] = h0_cell_val

        #mapping from visual space to word space
        if not hasattr(self, 'wvm'):
            wvm_val = init_layer_k(self.conf['visual_size'], self.conf['emb_size'])
            shared_layers['wvm'] = wvm_val

        if not hasattr(self, 'bmv'):
            bmv_val = np.zeros((self.conf['emb_size'],), dtype=theano.config.floatX)
            shared_layers['bmv'] = bmv_val

        #LSTM layer parameters
        if not hasattr(self, 'w_lstm'):
            w_lstm_val = init_layer_k(self.conf['lstm_hidden_size']*2, self.conf['lstm_hidden_size']*4)
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
                b_val[w2i[RNNDataProvider.STOP_TOKEN]] = np.log(self.X_train.shape[0]/float(wcount))
                for w in w2i:
                    if w in w2c:
                        b_val[w2i[w]] = np.log(w2c[w]/float(wcount))
                b_val -= np.max(b_val[1:])
            shared_layers['b'] = b_val

        self.build_shared_layers(shared_layers)

        #input variables for training
        self.x = T.imatrix("x")
        self.v = T.matrix("v")
        self.xlen = T.matrix("xlen")

        #input variables for generation
        self.v_single = T.vector("v")
        self.nstep = T.iscalar("nstep")

        #the dropout masks
        self.x_drop = T.tensor3("x_drop") #drop the input
        self.y_drop = T.tensor3("y_drop") #drop the output

        self.forced_word = T.imatrix("forced_word")

        h_tm1 = T.vector("h_tm1")       #hidden layer ouput
        word_t = T.ivector("word_t")    #word indexes
        v_i = T.vector("v")             #visual information

        #Generates the next word based on the: previous true word, hidden state & visual features
        #inputs: hiddent_layer, last_predicted word, visual features
        def recurrance(word_t, x_drop_slice, hh_drop_slice, use_v, h_tm1_hidden, h_tm1_cell, v_i):

                #word_t = theano.printing.Print("word_t")(word_t)

                #get the word embedding matrix or the context information
                if self.conf['DECODER']:
                    x_t = ifelse(T.eq(use_v, 1), T.dot(v_i, self.wvm) + self.bmv, self.wemb[word_t])
                else:
                    x_t = ifelse(T.eq(use_v, 1), T.zeros_like(self.wemb[word_t]), self.wemb[word_t])


                #if we are not doing minibatch training
                if word_t.ndim == 0:
                    x_t = x_t.reshape((1, x_t.shape[0]))
                    h_tm1_hidden = h_tm1_hidden.reshape((1, h_tm1_hidden.shape[0]))
                    h_tm1_cell = h_tm1_cell.reshape((1, h_tm1_cell.shape[0]))

                
                #dropout on the input embddings
                if self.conf['DROP_INPUT']:
                    x_t *= x_drop_slice

                #clip the gradients so they dont get too large 
                h_tm1_hidden_clip = self.clipg(h_tm1_hidden)

                in_state = T.concatenate([x_t, h_tm1_hidden_clip], axis=1)

                if self.conf['BATCH_NORM']:
                    mu = T.mean(in_state, axis=0, keepdims=True)
                    var = T.var(in_state, axis=0, keepdims=True)
                    normed_is = (in_state - mu) / T.sqrt(var + T.constant(1e-10, dtype=theano.config.floatX))
                    in_state = self.gamma_h * in_state + self.beta_h

                #calculate 8 dot products in one go 
                dot_out = T.dot(in_state, self.w_lstm)

                lstm_hidden_size = self.conf['lstm_hidden_size']
                #input gate 
                ig = T.nnet.sigmoid(dot_out[:, :lstm_hidden_size])
                #forget gate
                fg = T.nnet.sigmoid(dot_out[:, lstm_hidden_size:lstm_hidden_size*2])
                #output gate
                og = T.nnet.sigmoid(dot_out[:, lstm_hidden_size*2:lstm_hidden_size*3])

                # cell memory
                cc = fg * h_tm1_cell + ig * T.tanh(dot_out[:, lstm_hidden_size*3:])

                # hidden state
                hh = og * cc

                #drop the output state
                if self.conf['DROP_OUTPUT']:
                    hh_d = hh * hh_drop_slice

                #the distribution over output words
                if self.conf['SOFTMAX_OUT']:
                    s_t = T.nnet.softmax(T.dot(hh_d, self.w) + self.b)
                else:
                    s_t = T.nnet.sigmoid(T.dot(hh_d, self.w) + self.b)

                #hh = ifelse(T.eq(word_t, 0) and T.eq(use_v, 0), h_tm1_hidden, hh)
                #cc = ifelse(T.eq(word_t, 0) and T.eq(use_v, 0), h_tm1_cell, cc)

                if not self.conf['DECODER']:
                    keep_idx = T.and_(T.eq(word_t, 0),T.eq(use_v, 0))
                    #keep_idx = theano.printing.Print("keep_idx")(keep_idx)
                    if word_t.ndim != 0:
                        keep_idx = keep_idx.dimshuffle((0, 'x'))
                    #hh_ret = hh
                    #hh_ret[keep_idx, :] = h_tm1_hidden[keep_idx, :]
                    hh_ret = keep_idx * h_tm1_hidden + (1-keep_idx) * hh
                    cc_ret = keep_idx * h_tm1_cell + (1-keep_idx) * cc
                else:
                    hh_ret = hh
                    cc_ret = cc



                #if we are not doing minibatch training
                if word_t.ndim == 0:
                    hh_ret = hh_ret[0]
                    cc_ret = cc_ret[0]

                return [hh_ret, cc_ret, s_t]

        #Generates the next word by feeding the old word as input
        #inputs: hiddent_layer, last_predicted word, visual features
        def recurrance_word_feedback(h_tm1_hidden, h_tm1_cell, word_t, use_visual, v_i):
            x_drop_val = T.ones( (self.conf['emb_size'],) ,dtype=theano.config.floatX)
            y_drop_val = T.ones( (self.conf['lstm_hidden_size'],), dtype=theano.config.floatX)
            [hh, cc, s_t] = recurrance(word_t, x_drop_val, y_drop_val, use_visual, h_tm1_hidden, h_tm1_cell, v_i)
            
            #the predicted word
            w_idx = T.cast(T.argmax(s_t, axis=1), dtype='int32')[0]

            return [hh, cc, s_t[0], w_idx, T.zeros((0,), dtype='int32')[0]]

        def recurrance_partial_word_feedback(word_t_real, x_drop_val, y_drop_val, use_visual, forced_word, h_tm1_hidden, h_tm1_cell, word_t_pred, v_i):
            word_last = T.switch(forced_word, word_t_real, word_t_pred)
            [hh, cc, s_t] = recurrance(word_last, x_drop_val, y_drop_val, use_visual, h_tm1_hidden, h_tm1_cell, v_i)
            
            #the predicted word
            w_idx = T.cast(T.argmax(s_t, axis=1), dtype='int32')

            return [hh, cc, s_t, w_idx]


        #build the teacher forcing loop
        use_visual_info = T.concatenate([T.ones((1,), dtype=np.int32), T.zeros((self.conf['MAX_SENTENCE_LEN'],), dtype=np.int32)])
        if self.conf['DECODER']:
            #h0_hidden_matrix = self.encoder.hh_out[self.encoder.conf['MAX_SENTENCE_LEN']]

            h0_hidden_matrix = self.h0_hidden * T.ones((self.x.shape[0], self.h0_hidden.shape[0]))
            v_input = T.concatenate([self.encoder.hh_out[self.encoder.conf['MAX_SENTENCE_LEN']], self.v], axis=1)
            #v_input = T.printing.Print("v_input")(v_input)
        else:
            h0_hidden_matrix = self.h0_hidden * T.ones((self.x.shape[0], self.h0_hidden.shape[0]))
            v_input = self.v
            #v_input = T.printing.Print("v_input_v")(v_input)
            
        h0_cell_matrix = self.h0_cell * T.ones((self.x.shape[0], self.h0_cell.shape[0]))
        x_adj = T.concatenate([T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype), self.x.T])
        y_adj = T.concatenate([self.x.T, T.zeros((1, self.x.T[0].shape[0]), dtype=self.x.dtype)])
        [self.hh_out, self.cc_out, s], _ = theano.scan(fn=recurrance, 
                                sequences=[x_adj, self.x_drop.dimshuffle((1, 0, 2)), self.y_drop.dimshuffle((1, 0, 2)), use_visual_info],
                                n_steps=self.conf['MAX_SENTENCE_LEN']+1,
                                non_sequences=v_input, 
                                outputs_info=[h0_hidden_matrix, h0_cell_matrix, None])

        #build the semi-forced loop
        [_, _, s_semi, _], _ = theano.scan(fn=recurrance_partial_word_feedback, 
                                sequences=[x_adj, self.x_drop.dimshuffle((1, 0, 2)), self.y_drop.dimshuffle((1, 0, 2)), 
                                    use_visual_info, self.forced_word[:, :self.x.shape[0]]],
                                n_steps=self.conf['MAX_SENTENCE_LEN']+1,
                                non_sequences=self.v, 
                                outputs_info=[h0_hidden_matrix, h0_cell_matrix, None, T.zeros((self.x.shape[0],), dtype=np.int32)])

        #build the un-forced loop
        [_, _ , _ , self.wout_fb, _], _ = theano.scan(fn=recurrance_word_feedback, 
                                     non_sequences=self.v_single, 
                                     outputs_info=[self.h0_hidden, self.h0_cell, None, np.array(0, dtype=np.int32), T.ones((1,), dtype=np.int32)[0]],
                                        n_steps=self.nstep)

        if self.conf['SEMI_FORCED'] < 1:
            s = s_semi

        self.new_s = s.reshape((s.shape[0] * s.shape[1], s.shape[2]))
        softmax_out = self.build_loss_function(self.new_s, y_adj)
        self.softmax_out = softmax_out

        #calculate the perplexity
        ff_small = T.constant(1e-20, dtype=theano.config.floatX)
        ppl_idx = softmax_out.shape[1] * T.arange(softmax_out.shape[0]) + T.flatten(y_adj)
        hsum = -T.log2(T.flatten(softmax_out)[ppl_idx] + ff_small) 
        hsum_new = hsum.reshape((s.shape[0], s.shape[1])).T
        self.perplexity_sentence = 2 ** (T.sum(hsum_new, axis = 1) / T.sum(self.xlen, axis=1))
        self.perplexity_batch = 2 ** (T.sum(hsum * T.flatten(self.xlen.T)) / T.sum(self.xlen))
        self.perplexity_batch_v = T.sum(hsum * T.flatten(self.xlen.T))
        self.perplexity_batch_n = T.sum(self.xlen)

        #build the single step code
        h_hid = T.vector("h_hid")
        h_cell = T.vector("h_cell")
        x_drop_val = T.ones( (self.conf['emb_size'],) ,dtype=theano.config.floatX)
        y_drop_val = T.ones( (self.conf['lstm_hidden_size'],), dtype=theano.config.floatX)
        use_v = T.iscalar("use_v")
        word_t_s = T.iscalar("word_t_s")
        one_step_theano = recurrance(word_t_s, x_drop_val, y_drop_val, use_v, h_hid, h_cell, v_i)

        if self.conf['DECODER']:
            self.one_step = theano.function([word_t_s, use_v, h_hid, h_cell, v_i], outputs=one_step_theano)
        else:
            tmp_x = T.imatrix("tmp_x")
            tmp_v = T.matrix("tmp_v")
            x_d_tmp = T.ones( (1, self.conf['MAX_SENTENCE_LEN'], self.conf['emb_size']) ,dtype=theano.config.floatX)
            y_d_tmp = T.ones( (1, self.conf['MAX_SENTENCE_LEN'], self.conf['lstm_hidden_size']), dtype=theano.config.floatX)
            x_d_tmp.type.broadcastable = (False, False, False)
            y_d_tmp.type.broadcastable = (False, False, False)
            self.start_step = theano.function([tmp_x, tmp_v], 
                                                outputs=self.hh_out[self.conf['MAX_SENTENCE_LEN']],
                                                givens={self.x_drop : x_d_tmp,
                                                        self.y_drop : y_d_tmp,
                                                        self.x : tmp_x,
                                                        self.v : tmp_v})

    def build_loss_function(self, distributions, y_adj):
        new_s = distributions

        if self.conf['JOINED_LOSS_FUNCTION']:
            #note: we need to use the re-ordered output of the lm (ie self.lm_new_s) this accounts for dictionary differences
            #output_vocab_len = new_s.shape[1] / 2
            #weighted_results = self.new_s[:, :output_vocab_len] * self.mm_rnn.new_s + (1.0 - self.new_s[:, :output_vocab_len]) * self.lm_new_s
            #self.new_s = T.ones_like(self.new_s) * 0.90
            sm_res = self.new_s * self.mm_rnn.new_s + (1.0 - self.new_s) * self.lm_new_s
            #sm_res = self.mm_rnn.new_s
            sm_res = sm_res / T.sum(sm_res, axis=1, keepdims=True)
            #sm_res = T.nnet.softmax(weighted_results)
            #sm_res = self.lm_new_s
            loss_vec = T.nnet.categorical_crossentropy(sm_res, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            self.loss = T.sum(loss_vec) 
            return sm_res
        else:
            #calculate the total loss for this minibatch
            loss_vec = T.nnet.categorical_crossentropy(new_s, T.flatten(y_adj)) * T.flatten(self.xlen.T)
            self.loss = T.sum(loss_vec)
            return new_s

    #compile the functions needed to train the model
    def build_model_trainer(self):
        self.X_sh_train_mask = theano.shared(name="X_sh_train_mask", value=self.X_train_mask, borrow=True)
        self.X_sh_train = theano.shared(name="X_sh_train",value=self.X_train, borrow=True)
        self.V_sh_train = theano.shared(name="V_sh_train",value=self.V_train, borrow=True)
        if self.conf["DECODER"]:
            self.X_sh_train_drop = theano.shared(name="X_sh_train_drop",value=self.X_train_drop, borrow=True)
            self.Y_sh_train_drop = theano.shared(name="Y_sh_train_drop",value=self.Y_train_drop, borrow=True)
        if self.conf['JOINED_MODEL']:
            self.X_sh_train_lm = theano.shared(name="X_sh_train_lm",value=self.X_train_lm, borrow=True)

    
        params_train = [getattr(self,p) for p in self.conf['param_names_trainable']]

        #build the list of masks (which select which rows may be backpropagated)
        params_bp_mask = []
        for name in self.conf['param_names_trainable']:
            if name in self.conf['params_bp_mask']:
                params_bp_mask.append(self.conf['params_bp_mask'][name])
            else:
                params_bp_mask.append(None)
            
        if self.conf["DECODER"]:
            encoder_params = [getattr(self.encoder, p) for p in self.encoder.conf['param_names_trainable']]
            params_train = params_train + encoder_params
            for name in self.encoder.conf['param_names_trainable']:
                if name in self.encoder.conf['params_bp_mask']:
                    params_bp_mask.append(self.encoder.conf['params_bp_mask'][name])
                else:
                    params_bp_mask.append(None)


        #storage for historical gradients
        if not self.loaded_model or (not hasattr(self,'hist_grad') and not hasattr(self,'delta_grad')):
            self.hist_grad = [theano.shared(value=np.zeros_like(var.get_value()), borrow=True) for var in params_train]
            self.delta_grad = [theano.shared(value=np.zeros_like(var.get_value()), borrow=True) for var in params_train]

        if not self.conf["DECODER"]:
            return

        #calculate the cost for this minibatch (add L2 reg to loss function)
        regc = T.constant(self.conf['L2_REG_CONST'], dtype=theano.config.floatX)
        self.cost = self.loss + regc * np.sum(map(lambda xx: (xx ** 2).sum(), params_train)) 

        #build the SGD weight updates
        batch_size_f = T.constant(self.conf['batch_size_val'], dtype=theano.config.floatX)
        comp_grads = T.grad(self.cost, params_train)
        #if self.conf['DECODER']:
            #comp_grads[9] = T.printing.Print("Comp_grads_9")(comp_grads[9]) + 0.0000001*T.printing.Print("params_train_9")(params_train[9])
        comp_grads = [g/batch_size_f for g in comp_grads]
        comp_grads = [T.clip(g, -self.conf['GRAD_CLIP_SIZE'], self.conf['GRAD_CLIP_SIZE']) for g in comp_grads]
        #comp_grads = [g*m if m is not None else g for g,m in zip(comp_grads, params_bp_mask) ]
        weight_updates = get_sgd_weight_updates(self.conf['GRAD_METHOD'], comp_grads, params_train, self.hist_grad, self.delta_grad, 
                                        decay=self.conf['DECAY_RATE'], learning_rate=self.conf['LEARNING_RATE'])
        print "Weight updates:", len(weight_updates)
        #if self.conf['DECODER']:
        #    weight_updates[9] = (weight_updates[9][0], T.printing.Print("Comp_grads_9")(weight_updates[9][1]))

        indx = T.iscalar("indx")
        indx_wrap = indx % (self.X_sh_train_drop.shape[0] - self.conf['batch_size_val'])
        indx_wrap2 = (indx+1) % (self.X_sh_train_drop.shape[0] - self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            self.train = theano.function([indx], 
                        outputs=[self.loss, self.cost, self.perplexity_batch], 
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
                            self.lm_rnn.v: np.ones((self.conf['batch_size_val'], 1), dtype=theano.config.floatX),#self.V_sh_train[indx:indx+self.conf['batch_size_val']], 
                            self.lm_rnn.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                            self.lm_rnn.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                            self.lm_rnn.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']]
                            }, 
                        on_unused_input='ignore')

        else:
            if self.conf['SEMI_FORCED'] < 1:
                inputs = [indx, self.forced_word]
            else:
                inputs = [indx]
            if self.conf['DECODER']:
                print len(comp_grads)
                print len(params_train)
                print weight_updates
                self.train = theano.function(inputs, 
                            outputs=[self.loss, self.cost, self.perplexity_batch], 
                            updates=weight_updates, 
                            givens={
                                self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
                                self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']], 
                                self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                                self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                                self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
                                self.encoder.x: self.encoder.X_sh_train[indx:indx+self.conf['batch_size_val']],
                                self.encoder.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']],
                                self.encoder.xlen: self.encoder.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
                                self.encoder.x_drop: self.X_sh_train_drop[indx_wrap2:indx_wrap2+self.conf['batch_size_val']],
                                self.encoder.y_drop: self.Y_sh_train_drop[indx_wrap2:indx_wrap2+self.conf['batch_size_val']]}, 
                            on_unused_input='ignore')
#            else:
#                self.train = theano.function(inputs, 
#                            outputs=[self.loss, self.cost, self.perplexity_batch], 
#                            updates=weight_updates, 
#                            givens={
#                                self.x: self.X_sh_train[indx:indx+self.conf['batch_size_val']],
#                                self.v: self.V_sh_train[indx:indx+self.conf['batch_size_val']], 
#                                self.xlen: self.X_sh_train_mask[indx:indx+self.conf['batch_size_val']],
#                                self.x_drop: self.X_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']],
#                                self.y_drop: self.Y_sh_train_drop[indx_wrap:indx_wrap+self.conf['batch_size_val']]}, 
#                            on_unused_input='ignore')
        if not self.conf['DECODER']:
            return None

        
        return self.train

    def build_perplexity_calculator(self):
        mX, mY = self.make_drop_masks_identity(self.conf['batch_size_val'])
        if self.conf['JOINED_MODEL']:
            v_in = T.matrix("v_in")
            givens={self.v : v_in,
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
                    self.lm_rnn.xlen : self.xlen,
                    self.forced_word : T.ones((self.x.shape[1]+1, self.x.shape[0]), dtype=np.int32)}
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
            if self.conf['DECODER']:
                givens={self.x_drop: mX[:self.x.shape[0]], 
                        self.y_drop : mY[:self.x.shape[0]],
                        self.encoder.x_drop: mX[:self.x.shape[0]], 
                        self.encoder.y_drop : mY[:self.x.shape[0]],
                        self.forced_word : T.ones((self.x.shape[1]+1, self.x.shape[0]), dtype=np.int32)}
            else:
                givens={self.x_drop: mX[:self.x.shape[0]], 
                        self.y_drop : mY[:self.x.shape[0]],
                        self.forced_word : T.ones((self.x.shape[1]+1, self.x.shape[0]), dtype=np.int32)}
            self.get_ppl_val = theano.function(
                                    [self.x, self.v, self.xlen, self.encoder.x, self.encoder.v, self.encoder.xlen], 
                                    outputs=[self.perplexity_batch_v, self.perplexity_batch_n], 
                                    givens = givens,
                                    on_unused_input='ignore')
            #self.get_ppl_sent_val = theano.function(
            #                        [self.x, self.v, self.xlen], 
            #                        outputs=[self.perplexity_sentence], 
            #                        givens=givens,
            #                        on_unused_input='ignore')

    def build_sentence_generator(self):
        self.gen_sentence = theano.function([self.v_single, self.nstep], outputs=self.wout_fb)

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
                if self.conf['JOINED_MODEL']:
                    self.X_sh_train.set_value(self.X_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.V_sh_train.set_value(self.V_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.X_sh_train_mask.set_value(self.X_sh_train_mask.get_value(borrow=True)[idx], borrow=True)
                    self.X_sh_train_lm.set_value(self.X_sh_train_lm.get_value(borrow=True)[idx], borrow=True)
                    self.make_new_train_drop_masks()
                if self.conf['DECODER']:
                    self.X_sh_train.set_value(self.X_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.V_sh_train.set_value(self.V_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.X_sh_train_mask.set_value(self.X_sh_train_mask.get_value(borrow=True)[idx], borrow=True)
                    self.encoder.X_sh_train.set_value(self.encoder.X_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.encoder.V_sh_train.set_value(self.encoder.V_sh_train.get_value(borrow=True)[idx], borrow=True)
                    self.encoder.X_sh_train_mask.set_value(self.encoder.X_sh_train_mask.get_value(borrow=True)[idx], borrow=True)
                    #self.encoder.make_new_train_drop_masks()
                    self.make_new_train_drop_masks()
            if self.conf['SEMI_FORCED'] < 1:
                sf = np.array(np.random.binomial(1, self.conf['SEMI_FORCED'], size=(self.conf['MAX_SENTENCE_LEN']+1,batch_size_val)), dtype=np.int32)
                tr = self.train(cur_idx, sf)
            else:
                #print cur_idx
                #print self.encoder.X_sh_train.get_value().shape
                #print self.X_sh_train.get_value().shape
                #print self.encoder.X_sh_train.get_value()[cur_idx:cur_idx+self.conf['batch_size_val']]
                #print self.X_sh_train.get_value()[cur_idx:cur_idx+self.conf['batch_size_val']]
                #print self.encoder.V_sh_train.get_value()[cur_idx:cur_idx+self.conf['batch_size_val']][:6, :]
                #print self.V_sh_train.get_value()[cur_idx:cur_idx+self.conf['batch_size_val']][:6, :]
                tr = self.train(cur_idx)
                #sys.exit(0)
            #print self.encoder.w_lstm.get_value()
            print tr, num_iter * batch_size_val / float(num_train_examples)

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
                if self.conf['DECODER']:
                    ppl_v, ppl_n = self.get_ppl_val(self.X_valid[ii:ii+batch_size_val], 
                                                self.V_valid[ii:ii+batch_size_val], 
                                                self.X_valid_mask[ii:ii+batch_size_val],
                                                self.encoder.X_valid[ii:ii+batch_size_val],
                                                self.encoder.V_valid[ii:ii+batch_size_val], 
                                                self.encoder.X_valid_mask[ii:ii+batch_size_val])
                    for idx in xrange(ii, ii+batch_size_val):
                        xt = self.encoder.X_valid[idx]
                        for x in xt:
                            if x == 0:break
                            print self.encoder.model['i2w'][x],
                        print ""
                        xt = self.X_valid[idx]
                        for x in xt:
                            if x == 0:break
                            print self.model['i2w'][x],
                        print "\n"
                    sys.exit(0)
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

        ppl = self.get_ppl_sent_val(x_pad, v, x_len_mask)
        return ppl
    
    def do_one_step(self, v_i, last_step = None, encoder_words = None):
        if last_step is not None:
            step = last_step
        else:
            step = {'word_t': 0,
                    'h_hid': np.zeros(self.conf['lstm_hidden_size'], dtype=theano.config.floatX),
                    'h_cell': np.zeros(self.conf['lstm_hidden_size'], dtype=theano.config.floatX),
                    'use_v': np.array(1, dtype=np.int32)}
            if self.conf['DECODER']:
                hh = self.encoder.start_step(np.reshape(encoder_words, (1, -1)), np.reshape(v_i, (1, -1)))[0]
                #v_i = hh
                v_i = np.concatenate([hh, v_i], axis=1)

        hh, cc, s_t = self.one_step(step['word_t'], step['use_v'], step['h_hid'], step['h_cell'], v_i)

        step['h_hid'] = hh
        step['h_cell'] = cc
        step['s_t'] = s_t
        step['word_t'] = np.argmax(s_t)
        step['use_v'] = np.array(0, dtype=np.int32)
        return step


    #generate a sentence by sampling from the predicted distributions
    def sample_sentence(self, v, encoder_words=None, MAP=False):
        sentence = []
        last_step = None
        for i in xrange(self.conf['MAX_SENTENCE_LEN'] + 1):
            last_step = self.do_one_step(v, last_step, encoder_words)
            c = np.arange(last_step['s_t'][0].shape[0])
            p = np.array(last_step['s_t'][0], dtype=np.float64)
            p = p / p.sum()
            if MAP:
                w_i = np.argmax(p)
            else:
                w_i = np.random.choice(c, p=p)
            if w_i == 0:
                break
            sentence.append(self.model['i2w'][w_i])
        return sentence
    
    def sentence_idx_to_str(self, sen):
        sentence = []
        i2w = self.model['i2w']
        for i in sen:
            if i == 0:break
            sentence.append(i2w[i])
        return sentence

    def get_sentence(self, v):
        res = self.gen_sentence(v, self.conf['MAX_SENTENCE_LEN'] + 1)
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
