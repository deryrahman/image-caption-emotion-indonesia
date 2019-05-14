import numpy as np
import theano
import json


#get all the conversions from one dictionary to another
def get_dictionary_a_to_b(w2i_a, i2w_a, w2i_b, i2w_b):
    a2b = {}
    a2b_vec = np.zeros((len(i2w_a),), dtype=np.int32)
    a2b_mask = np.zeros((len(i2w_a),), dtype=np.int32)
    for i, w in i2w_a.items():
        if w in w2i_b:
            a2b[i] = w2i_b[w]
            a2b_vec[i] = w2i_b[w]
            a2b_mask[i] = 1

    b2a = dict([(i2, i1) for i1, i2 in a2b.items()])
    return a2b, a2b_vec, a2b_mask, b2a


#turn a list of lists into a numpy array padded with zeros
def pad_vectors(vals):
    mlen = np.max([v.shape[0] for v in vals])
    r = []
    for v in vals:
        vv = np.array(v, dtype=np.int32)
        vv.resize((mlen,))
        r.append(vv)
    return np.array(r, dtype=np.int32)


def build_len_mask(Xlen, mask_size):
    masks = np.zeros((mask_size,), dtype=theano.config.floatX)
    masks[:Xlen] = 1.0
    return masks


def build_len_masks(Xlen, mask_size):
    masks = np.zeros((Xlen.shape[0], mask_size), dtype=theano.config.floatX)
    for i, l in enumerate(Xlen):
        masks[i, :l] = 1.0
    return masks


# karpathys layer initialisation
def init_layer(nrow, ncol):
    return np.array((np.random.rand(nrow, ncol) * 2 - 1) * 0.01,
                    dtype=theano.config.floatX)


def init_layer_k(nrow, ncol):
    return np.array((np.random.rand(nrow, ncol) * 2 - 1) * 0.1,
                    dtype=theano.config.floatX)


# layer initialisation based on blog about caffe initialisation
def init_layer_xavier(nrow, ncol, scale=1.0):
    var = 1.1 * 2.0 / (nrow + ncol)
    d = np.sqrt(3.0 * var)
    return np.array(np.random.uniform(-d, d, (nrow, ncol)) * scale,
                    dtype=theano.config.floatX)


def init_layer_xavier_1(nrow, ncol, scale=1.0):
    var = 2.0 / (nrow + ncol)
    d = np.sqrt(3.0 * var)
    return np.array(np.random.uniform(-d, d, (nrow, ncol)) * scale,
                    dtype=theano.config.floatX)


def make_drop_mask(ninst, nrow, ncol, drop_prob):
    return np.array(
        (np.random.rand(ninst, nrow, ncol) < drop_prob) * (1.0 / drop_prob),
        dtype=theano.config.floatX)


class GradClip(theano.compile.ViewOp):
    # See doc in user fct grad_clip
    __props__ = ()

    def __init__(self, clip_lower_bound, clip_upper_bound):
        # We do not put those member in __eq__ or __hash__
        # as they do not influence the perform of this op.
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert (self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [
            theano.tensor.clip(g_out, self.clip_lower_bound,
                               self.clip_upper_bound) for g_out in g_outs
        ]


def grad_clip(x, lower_bound, upper_bound):
    return GradClip(lower_bound, upper_bound)(x)


class GradReverse(theano.compile.ViewOp):
    __props__ = ()

    def __init__(self, c):
        self.c = c
        assert (c > 0)

    def grad(self, args, g_outs):
        return [-self.c * g_out for g_out in g_outs]


def grad_reverse(x, c=0.1):
    return GradReverse(c)(x)


class GradIgnore(theano.compile.ViewOp):
    __props__ = ()

    def grad(self, args, g_outs):
        return [0 * g_out for g_out in g_outs]


def grad_ignore(x):
    return GradIgnore()(x)


def save_captions_json(filename, img_files, captions):
    data = {"images": []}
    for img, capts in zip(img_files, captions):
        example = {}
        example['filename'] = img
        capts = [" ".join(c) if type(c) is list else c for c in capts]
        example['captions'] = capts

        data['images'].append(example)
    print data
    json.dump(data, open(filename, "w"))


def save_captions_annotated_json(filename, img_files, captions, annotations,
                                 anps_res, orig_sent):
    data = {"images": []}
    for img, capts, anna, anps, orig in zip(img_files, captions, annotations,
                                            anps_res, orig_sent):
        example = {}
        example['filename'] = img
        capts = [c.split() if type(c) is str else c for c in capts]
        example['captions'] = capts
        example['annotations'] = anna
        example['anps'] = anps
        example['orig'] = orig

        data['images'].append(example)
    print data
    json.dump(data, open(filename, "w"))
