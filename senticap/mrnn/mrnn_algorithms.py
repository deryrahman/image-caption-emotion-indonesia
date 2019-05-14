import copy
import numpy as np


class BeamInstance:

    def __init__(self):
        self.log_prob = 0.0
        self.oob_log_prob = 0.0
        self.word_idxs = []
        self.highlight = []
        self.rnn_state = None
        self.score = 0.0
        self.force_next = 0
        self.forced_count = 0
        self.forced_word = ""

    def get_per_word(self, quantity):
        wds = len(self.word_idxs) - self.forced_count
        if wds == 0: return 0.0
        return quantity / float(wds)

    def __lt__(self, other):

        lp_pw_a = self.get_per_word(self.log_prob)
        lp_pw_b = other.get_per_word(other.log_prob)
        return lp_pw_a < lp_pw_b

    def do_step(self, rnn, v, encoder_words=None):
        if encoder_words is not None:
            self.rnn_state = rnn.do_one_step(v, self.rnn_state, encoder_words)
        else:
            self.rnn_state = rnn.do_one_step(v, self.rnn_state)

    def done(self):
        if self.word_idxs and self.word_idxs[-1] == 'STOPSENTENCE':
            return True
        if len(self.word_idxs) == 20:
            return True
        return False

    def get_best_words(self, k):
        return np.argsort(-self.rnn_state['s_t'][0])[:k]

    def choose_word(self, w, i2w):
        was_forced = False
        word = i2w[w]
        if self.force_next != 0:
            w = self.force_next
            self.force_next = 0
            self.forced_count += 1
            was_forced = True
            word = self.forced_word
            self.forced_word = ""
            self.highlight.append(1)
        self.rnn_state['word_t'] = w
        if not was_forced:
            self.log_prob += -np.log(self.rnn_state['s_t'][0][w])
            self.score += self.rnn_state['s_t'][0][w]
            self.highlight.append(0)
        self.word_idxs.append(word)

    def finalize(self):
        self.log_prob = self.get_per_word(self.log_prob)
        self.score = self.get_per_word(self.score)

    def get_sentence_string(self):
        return self.word_idxs[:-1]

    def get_highlights(self):
        return self.highlight[:-1]


#use a beam search to decode the sentence
def decoder_beamsearch2(rnn, v, beam_size=20, with_score=False):
    results = []
    beam = [BeamInstance()]
    while len(beam) > 0:
        #advance all the states in the beam
        for b in beam:
            b.do_step(rnn, v)

        #get all the possible new states
        possible_states = []
        for b in beam:

            #store this as a finished beam
            if b.done():
                b.finalize()
                results.append(b)
                continue

            #try the top words
            best_words = b.get_best_words(beam_size)
            for w in best_words:
                new_b = copy.deepcopy(b)
                new_b.choose_word(w, rnn.model['i2w'])
                possible_states.append(new_b)

        #filter the possible states to create the new beam
        possible_states.sort()
        beam = possible_states[:beam_size]

    results.sort()

    if with_score:
        return results[0].score, results[0].get_sentence_string()

    return results[0].log_prob, results[0].get_sentence_string()


#use a beam search to decode the sentence
def decoder_beamsearch(rnn, v, senti=None, beam_size=20, with_score=False):
    results = []
    if senti is not None:
        beam = [(0.0, 0, [], rnn.do_one_step(v, senti=senti,
                                             last_step=None), 0)]
    else:
        beam = [(0.0, 0, [], rnn.do_one_step(v, None), 0)]
    while len(beam) > 0:
        new_beam = []
        for lp, c, w_idx, b, s in beam:
            #try the best new words up to beam_size
            all_lp = -np.log2(b['s_t'][0])
            score = b['s_t'][0]
            all_lp_srt = np.argsort(all_lp)
            for i in all_lp_srt[:beam_size]:
                b['word_t'] = i
                if i == 0 or c == 20:  #got the end token
                    results.append(
                        ((all_lp[i] + lp) / (c + 1), c + 1, w_idx + [i],
                         copy.deepcopy(b), s + score[i]))
                    #results.append(((all_lp[i] + lp), c+1, w_idx + [i], copy.deepcopy(b), s+score[i]))
                elif c < 20:
                    new_beam.append((all_lp[i] + lp, c + 1, w_idx + [i],
                                     copy.deepcopy(b), s + score[i]))
        beam = sorted(new_beam, key=lambda x: x[0] / x[1])[:beam_size]
        #beam = sorted(new_beam, key=lambda x: x[0])[:beam_size]
        beam = [(lp, c, w_idx, rnn.do_one_step(v, b), s)
                for lp, c, w_idx, b, s in beam]

    results.sort()

    lp = 2**-results[0][0]
    sen_idx = results[0][2]
    sen = [rnn.model['i2w'][sidx] for sidx in sen_idx]

    if with_score:
        return results[0][4] / results[0][1], sen[:-1]
        des_sen = decoder_beamsearch(rnn,
                                     rnn.V_valid[idx],
                                     senti=-1.0,
                                     beam_size=1)[1][:-1]
    return lp, sen


#use a beam search to decode the sentence
def decoder_beamsearch_with_attention(rnn,
                                      v,
                                      senti=None,
                                      beam_size=20,
                                      with_score=False):
    results = []
    if senti is not None:
        beam = [(0.0, 0, [], rnn.do_one_step(v, senti=senti,
                                             last_step=None), 0, [])]
    else:
        beam = [(0.0, 0, [], rnn.do_one_step(v, None), 0, [])]
    while len(beam) > 0:
        new_beam = []
        for lp, c, w_idx, b, s, att in beam:
            #try the best new words up to beam_size
            all_lp = -np.log2(b['s_t'][0])
            score = b['s_t'][0]
            #print "Attout", b['att_out']
            all_lp_srt = np.argsort(all_lp)
            for i in all_lp_srt[:beam_size]:
                att_new = copy.deepcopy(att)
                b['word_t'] = i
                if i == 0 or c == 20:  #got the end token
                    att_new.append(b['att_out'][0][0])
                    results.append(
                        ((all_lp[i] + lp) / (c + 1), c + 1, w_idx + [i],
                         copy.deepcopy(b), s + score[i], att_new))
                    #results.append(((all_lp[i] + lp), c+1, w_idx + [i], copy.deepcopy(b), s+score[i]))
                elif c < 20:
                    att_new.append(b['att_out'][0][0])
                    new_beam.append((all_lp[i] + lp, c + 1, w_idx + [i],
                                     copy.deepcopy(b), s + score[i], att_new))
        beam = sorted(new_beam, key=lambda x: x[0] / x[1])[:beam_size]
        #beam = sorted(new_beam, key=lambda x: x[0])[:beam_size]
        beam = [(lp, c, w_idx, rnn.do_one_step(v, b), s, att)
                for lp, c, w_idx, b, s, att in beam]

    results.sort()

    lp = 2**-results[0][0]
    sen_idx = results[0][2]
    sen = [rnn.model['i2w'][sidx] for sidx in sen_idx]
    att = results[0][5]

    if with_score:
        return results[0][4] / results[0][1], sen[:-1]
        des_sen = decoder_beamsearch(rnn,
                                     rnn.V_valid[idx],
                                     senti=-1.0,
                                     beam_size=1)[1][:-1]
    return lp, sen, att
