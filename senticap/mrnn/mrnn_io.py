import numpy as np
import string
import scipy
import scipy.io
import json
import theano
import pickle

DO_NEG = False


def parse_word(w):
    w = w.encode('utf-8')
    w = str.lower(w)
    w = w.translate(string.maketrans("", ""), string.punctuation)
    return w


def pad_vector(vec, max_pad_len):
    vv = np.array(vec[:max_pad_len], dtype=np.int32)
    vv.resize((max_pad_len,))
    return vv


#turn a list of lists into a numpy array padded with zeros
def pad_vectors(vals, max_pad_len):
    r = []
    for v in vals:
        vv = pad_vector(v, max_pad_len)
        r.append(vv)
    return np.array(r, dtype=np.int32)


class RNNDataProvider:

    COCO = "coco"
    COCO_EXTRA = "coco_extra"
    COCO_MTURK = "coco_mturk"
    COCO_MTURK_WCOCO = "coco_mturk_wcoco"
    FLK8 = "flk8"
    FLK8LM = "flk8lm"
    FLK30 = "flk30"
    FLK30LM = "flk30lm"
    FLK30LM_SENT = "flk30lm_sent"
    FLK30LM_PART = "flk30lm_part"
    YH100LM = "yh100lm"

    TRAIN = 1
    TEST = 2
    VAL = 4
    TEST_VAL = 6

    START_TOKEN = "STARTSENTENCE"
    STOP_TOKEN = "STOPSENTENCE"

    MIN_WORD_FREQ = 5

    def read_dataset_lm(self):
        self.loaded = True
        self.tokens = []
        self.split = []
        self.img_id = []
        self.tok_to_feat = []
        self.img_filename = []
        data = pickle.load(open(self.DATASET_FILE_DATA, 'r'))
        np.random.seed(1234)
        np.random.shuffle(data)
        #np.random.seed()
        train_part = (len(data) / 10) * 8

        self.tok_to_feat = []
        self.feat_to_tok = []
        num_tok = 0

        seen_sentences = set()
        for i, sen in enumerate(data):
            sentence = []
            if type(sen) is str:
                sen = sen.split()
            for w in sen:
                w = parse_word(w)
                if not w:
                    continue
                sentence.append(w)

            if i < train_part:
                splt = self.TRAIN
            else:
                splt = self.VAL

            if self.reverse:
                sentence = sentence[::-1]

            sen_join = " ".join(sentence)
            if sen_join in seen_sentences:
                continue
            seen_sentences.add(sen_join)

            if sentence:
                self.tokens.append(sentence)
                self.split.append(splt)
                self.tok_to_feat.append(num_tok)
                self.feat_to_tok.append([num_tok])
                self.img_id.append(i)
                num_tok += 1
        print("Sentences: ", len(self.tokens))

    #read the dataset
    def read_dataset_mm(self):
        self.loaded = True
        self.tokens = []
        self.split = []
        self.img_id = []
        self.tok_to_feat = []
        self.img_filename = []
        self.img_id_to_filename = {}

        js = json.load(open(self.DATASET_FILE_DATA, "r"))

        self.feat_to_tok = [[] for i in range(len(js["images"]))]
        num_tok = 0
        for i, img in enumerate(js["images"]):
            if img["split"] == "train":
                splt = self.TRAIN
            elif img["split"] == "test":
                splt = self.TEST
            elif img["split"] == "val":
                splt = self.VAL

            for sen in img["sentences"]:
                sentence = []
                for w in sen["tokens"]:
                    w = parse_word(w)
                    if not w:
                        continue
                    sentence.append(w)
                if self.reverse:
                    sentence = sentence[::-1]

                if sentence:
                    self.tokens.append(sentence)
                    self.split.append(splt)
                    self.img_id.append(img["imgid"])
                    self.img_filename.append(img["filename"])
                    self.img_id_to_filename[img["imgid"]] = img["filename"]
                    self.tok_to_feat.append(i)
                    self.feat_to_tok[i].append(num_tok)
                    num_tok += 1

    #read the dataset
    def read_dataset_mm_extra(self, max_size=10000):
        self.loaded = True
        self.tokens = []
        self.split = []
        self.img_id = []
        self.tok_to_feat = []
        self.img_filename = []
        self.img_id_to_filename = {}
        self.tokens_desc = []
        self.tokens_title = []
        self.tokens_tags = []

        js = json.load(open(self.DATASET_FILE_DATA, "r"))

        self.feat_to_tok = [[] for i in range(len(js["images"]))]
        num_tok = 0
        num_train = 0
        num_test = 0
        num_val = 0
        for i, img in enumerate(js["images"]):
            if img["split"] == "train":
                splt = self.TRAIN
                if num_train >= max_size:
                    continue
                num_train += 1
            elif img["split"] == "test":
                splt = self.TEST
                if num_test >= max_size:
                    continue
                num_test += 1
            elif img["split"] == "val":
                splt = self.VAL
                if num_val >= max_size:
                    continue
                num_val += 1

            desc = []
            title = []
            tags = []
            for w in img['extra']["title"]:
                w = parse_word(w)
                if not w:
                    continue
                title.append(w)
            for w in img['extra']["desc"]:
                w = parse_word(w)
                if not w:
                    continue
                desc.append(w)
            tags = img['extra']["tags"]
            if self.reverse:
                title = title[::-1]
                desc = desc[::-1]

            for sen in img["sentences"]:
                sentence = []
                for w in sen["tokens"]:
                    w = parse_word(w)
                    if not w:
                        continue
                    sentence.append(w)

                if self.reverse:
                    sentence = sentence[::-1]

                if sentence:
                    self.tokens.append(sentence)
                    self.tokens_desc.append(desc)
                    self.tokens_title.append(title)
                    self.tokens_tags.append(tags)
                    self.split.append(splt)
                    self.img_id.append(img["imgid"])
                    self.img_filename.append(img["filename"])
                    self.img_id_to_filename[img["imgid"]] = img["filename"]
                    self.tok_to_feat.append(i)
                    self.feat_to_tok[i].append(num_tok)
                    num_tok += 1

    def read_dataset_mm_mturk(self):
        self.loaded = True
        self.tokens = []
        self.split = []
        self.img_id = []
        self.tok_to_feat = []
        self.img_filename = []
        self.img_id_to_filename = {}
        self.img_id_to_tokens = {}
        self.senti_words = []
        self.sentiment = []

        js = json.load(open(self.DATASET_FILE_DATA, "r"))

        self.feat_to_tok = [[] for i in range(len(js["images"]))]
        num_tok = 0
        for i, img in enumerate(js["images"]):
            if img["split"] == "train":
                splt = self.TRAIN
            elif img["split"] == "test":
                splt = self.TEST
            elif img["split"] == "val":
                splt = self.VAL

            for sen in img["sentences"]:
                sentence = []
                if DO_NEG:
                    if sen["sentiment"] != 0:
                        continue
                else:
                    if sen["sentiment"] != 1:
                        continue
                for w in sen["tokens"]:
                    w = parse_word(w)
                    if not w:
                        continue
                    sentence.append(w)
                if self.reverse:
                    sentence = sentence[::-1]

                if sentence:
                    self.tokens.append(sentence)
                    self.split.append(splt)
                    self.img_id.append(img["imgid"])
                    self.img_filename.append(img["filename"])
                    self.img_id_to_filename[img["imgid"]] = img["filename"]
                    if img["imgid"] not in self.img_id_to_tokens:
                        self.img_id_to_tokens[img["imgid"]] = []
                    self.img_id_to_tokens[img["imgid"]].append(
                        len(self.tokens) - 1)
                    self.tok_to_feat.append(i)
                    self.feat_to_tok[i].append(num_tok)
                    num_tok += 1
                    self.sentiment.append(sen["sentiment"])
                    if self.reverse:
                        self.senti_words.append(sen["word_sentiment"][::-1])
                    else:
                        self.senti_words.append(sen["word_sentiment"])

    def __init__(self, dataset_name, reverse=False):
        self.reverse = reverse
        self.loaded = False
        if dataset_name == self.FLK8:
            self.DATASET_FILE_FEATURES = "./flk8/flk8.mat"
            self.DATASET_FILE_DATA = "./flk8/flk8.json"
            self.read_dataset = self.read_dataset_mm
        elif dataset_name == self.FLK8LM:
            self.DATASET_FILE_FEATURES = ""
            self.DATASET_FILE_DATA = "./flk8/flk8.json"
            self.read_dataset = self.read_dataset_mm
        elif dataset_name == self.COCO:
            self.DATASET_FILE_FEATURES = "./coco/vgg_feats.mat"
            self.DATASET_FILE_DATA = "./coco/dataset.json"
            self.read_dataset = self.read_dataset_mm
        elif dataset_name == self.COCO_EXTRA:
            self.DATASET_FILE_FEATURES = "./coco/vgg_feats.mat"
            self.DATASET_FILE_DATA = "./coco_extra/dataset_extra.json"
            self.read_dataset = self.read_dataset_mm_extra
        elif dataset_name == self.COCO_MTURK:
            self.DATASET_FILE_FEATURES = "./coco/vgg_feats.mat"
            if DO_NEG:
                self.DATASET_FILE_DATA = "./coco_mturk/dataset_mturk_sentiment2_neg.json"
            else:
                self.DATASET_FILE_DATA = "./coco_mturk/dataset_mturk_sentiment2.json"
            self.read_dataset = self.read_dataset_mm_mturk
        elif dataset_name == self.COCO_MTURK_WCOCO:
            self.DATASET_FILE_FEATURES = "./coco/vgg_feats.mat"
            self.DATASET_FILE_DATA = "./coco_mturk/dataset_mturk_sentiment2_wcoco.json"
            self.read_dataset = self.read_dataset_mm_mturk
        elif dataset_name == self.FLK30LM:
            self.DATASET_FILE_FEATURES = ""
            self.DATASET_FILE_DATA = "./flk30_lm/flk30_not8k_sentences.pik"
            self.read_dataset = self.read_dataset_lm
        elif dataset_name == self.FLK30LM_SENT:
            self.DATASET_FILE_FEATURES = "./flk30_lm/flk30_sentiment.mat"
            self.DATASET_FILE_DATA = "./flk30_lm/flk30_not8k_sentences.pik"
            self.read_dataset = self.read_dataset_lm
        elif dataset_name == self.FLK30:
            self.DATASET_FILE_FEATURES = "./flickr30k/vgg_feats.mat"
            self.DATASET_FILE_DATA = "./flickr30k/dataset.json"
            self.read_dataset = self.read_dataset_mm
        elif dataset_name == self.FLK30LM_PART:
            self.DATASET_FILE_FEATURES = ""
            self.DATASET_FILE_DATA = "./flickr30k/dataset.json"
            self.read_dataset = self.read_dataset_mm
        elif dataset_name == self.YH100LM:
            self.DATASET_FILE_FEATURES = ""
            self.DATASET_FILE_DATA = "./yfcc100m/yahoo_100m_saved_sentences.pik"
            self.read_dataset = self.read_dataset_lm

    #read the context features
    def read_context(self):
        if self.DATASET_FILE_FEATURES:
            self.feats = scipy.io.loadmat(self.DATASET_FILE_FEATURES)["feats"]
            self.feats = self.feats.T
        else:
            self.feats = np.zeros((len(self.tokens), 1),
                                  dtype=theano.config.floatX)

    #count the number of times each word occurs in the dataset
    def get_word_counts(self, data_split, source="tokens"):
        # data_source = []
        # if source == "tokens":
        #     data_source = self.tokens
        # elif source == "title":
        #     data_source = self.tokens_title
        # elif source == "tags":
        #     data_source = self.tokens_tags
        # elif source == "desc":
        #     data_source = self.tokens_desc
        w_count = {}
        for i in range(len(self.tokens)):
            if self.split[i] != data_split:
                continue
            for w in self.tokens[i]:
                if w not in w_count:
                    w_count[w] = 0
                w_count[w] += 1
        return w_count

    #build a vocabulary from the training data
    def build_vocab(self, min_freq, source="tokens"):
        w_count = self.get_word_counts(self.TRAIN, source)

        w2idx = {}
        cur_vocab_indx = 1
        w2idx[self.START_TOKEN] = 0
        w2idx[self.STOP_TOKEN] = 0
        for w in w_count:
            if w_count[w] < min_freq:
                continue
            w2idx[w] = cur_vocab_indx
            cur_vocab_indx += 1

        self.w2i = w2idx
        self.i2w = dict([(v[1], v[0]) for v in list(w2idx.items())])

    #represent a sentences as indices from the vocabulary
    def tokenize_sentence(self, sentence):
        stok = []
        used = []
        for i, w in enumerate(sentence):
            if w in self.w2i:
                stok.append(self.w2i[w])
                used.append(i)
        return stok, np.array(used)

    #get each of the data splits
    def get_data_split(self,
                       data_split,
                       randomize=False,
                       rotate_X_with_id=False,
                       pad_len=20,
                       source="tokens",
                       anp_switch=False):
        data_source = []
        if source == "tokens":
            data_source = self.tokens
        elif source == "title":
            data_source = self.tokens_title
        elif source == "tags":
            data_source = self.tokens_tags
        elif source == "desc":
            data_source = self.tokens_desc
        X = []
        Xlen = []
        V = []
        Id = []
        anp_sw_pos = []
        sentiment = []
        X_pad = []
        for i in range(len(data_source)):
            if self.split[i] & data_split != self.split[i]:
                continue
            tok, used = self.tokenize_sentence(data_source[i])
            if anp_switch:
                pos_vec = np.array(np.zeros((pad_len + 1,)))

                for pos_pos in np.flatnonzero(np.array(self.senti_words[i])):
                    anp_pos_new = np.count_nonzero(used < pos_pos)
                    if np.any(used == pos_pos) and anp_pos_new < pad_len:
                        pos_vec[anp_pos_new] = 1

                #print self.senti_words[i]
                #print tok, pos_vec
                #for w, pv in zip(tok, pos_vec):
                #    print "%s_%d" % (self.i2w[w], pv),
                #print "\n"

                #if np.random.rand() < 0.01:
                #    sys.exit(0)

                #if self.sentiment[i] == 0:continue
                anp_sw_pos.append(pos_vec)
                sentiment.append(np.abs(self.sentiment[i] * 2 - 1))

            X.append(tok)
            Xlen.append(len(tok))

            V.append(self.feats[self.img_id[i]])
            Id.append(self.img_id[i])

        anp_sw_pos = np.array(anp_sw_pos, dtype=theano.config.floatX)
        sentiment = np.array(sentiment, dtype=theano.config.floatX)
        X_pad = pad_vectors(X, pad_len)
        Xlen = np.array(Xlen)
        V = np.array(V, dtype=theano.config.floatX)
        Id = np.array(Id)

        if rotate_X_with_id:
            idx_rotate = []
            cur_block = []
            idx_last = Id[0]
            for i, idx in enumerate(Id):
                if idx == idx_last:
                    cur_block.append(i)
                else:
                    for c in [cur_block[-1]] + cur_block[:-1]:
                        idx_rotate.append(c)
                    cur_block = [i]
                idx_last = idx
            if cur_block:
                for c in [cur_block[-1]] + cur_block[:-1]:
                    idx_rotate.append(c)
            idx_rotate = np.array(idx_rotate)
        else:
            idx_rotate = np.arange(Id.shape[0])

        print("Dataset Size:", Xlen.shape)
        idx = np.arange(X_pad.shape[0])
        if randomize:
            np.random.shuffle(idx)
        if anp_switch:
            return X_pad[idx_rotate[idx]], Xlen[
                idx_rotate[idx]], V[idx], Id[idx], anp_sw_pos[
                    idx_rotate[idx]], sentiment[idx_rotate[idx]]
        return X_pad[idx_rotate[idx]], Xlen[idx_rotate[idx]], V[idx], Id[idx]

    def make_single_data_instance(self, sentence, pad_len):
        if type(sentence) is str:
            sentence = sentence.split()
        tok, used = self.tokenize_sentence(sentence)
        tlen = len(tok)
        tok_pad = pad_vector(tok, pad_len)

        return tok_pad, tlen
