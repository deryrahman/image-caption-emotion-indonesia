import cPickle
import spacy
from mrnn.mrnn import *
from mrnn.mrnn_algorithms import *

#the reward we get for chosing a particular sentiment (given the goal)
def reward(goal, s):
    sig = 0.8
    a  = 1.0 / (sig * np.sqrt(2.0 * np.pi)) 
    b = - (goal - s)**2 / (2.0 * sig**2)
    r = a * np.exp(b)
    return r

#calculate the rewards for a set of (anp, sentiment) pairs
def get_anp_reward(goal, anp_to_senti):
    anp_to_reward = {}
    for w, s in anp_s:
        anp_to_reward[w] = reward(goal, s)
    return anp_to_reward

#the caffe scores for each ANP
class ANPVisualScores:
    def __init__(self, score_filename, id_filename):
        self.anp_pred = cPickle.load(open(score_filename, "rb"))
        self.anp_to_id = cPickle.load(open(id_filename, "rb"))
        
    def get_score(self, img_idx, anp):
        if anp not in self.anp_to_id: return 1.0 / len(self.anp_to_id)
        anp_id = self.anp_to_id[anp]
        score = self.anp_pred[img_idx][anp_id]
        return score

#the sentiment scores for each ANP
class SentiScores:
    def __init__(self, filename):
        self.noun_to_senti = cPickle.load(open(filename, "rb"))
        self.cache = {}

    def get_anp_to_score(self, all_nouns, goal=None):
        if goal in self.cache:
            return self.cache[goal]

        anps = []
        senti_scores = []
        n2s = []
        anps_to_s = {}
        for noun in all_nouns:
            if noun in self.noun_to_senti:
                n2s = self.noun_to_senti[noun]
            else:
                n2s = []
            n2s.append(('', 0.0))

            for adj, s in n2s:
                anp = adj + "_" + noun
                if goal is not None:
                    s = reward(goal, s)
                anps.append(anp)
                senti_scores.append(s)
                anps_to_s[anp] = s
        self.cache[goal] = anps_to_s
        return anps_to_s

    def get_anp_score_from_noun(self, noun, goal = None):
        anps = []
        senti_scores = []
        n2s = []
        if noun in self.noun_to_senti:
            n2s = self.noun_to_senti[noun]
            n2s.append(('', 0.0))
        else:
            return [], np.empty((1,))

        for adj, s in n2s:
            anp = adj + "_" + noun
            if goal is not None:
                s = reward(goal, s)
            anps.append(anp)
            senti_scores.append(s)
        return anps, np.array(senti_scores)

def anp_joint_score_prob(goal, w2i, noun_pd, img_idx, sentiscore, vscore, scores_from_noun, C, get_all_scores = False):

    nouns = set(w2i.keys())

    #calculate E(anp | s)
    anps_to_senti_score = sentiscore.get_anp_to_score(nouns, goal)

    #calculate E(anp | I)
    if C[0] == 0:
        anps_to_vis_score = dict([(anp, 1.0/len(anps_to_senti_score)) for anp in anps_to_senti_score])
    else:
        anps_to_vis_score = dict([(anp, vscore.get_score(img_idx, anp)) for anp in anps_to_senti_score])

    #calculate E(n | w, I)
    anps_to_noun_pd = {}
    all_nouns = set()
    for anp in anps_to_senti_score:
        noun = anp.split('_')[1]
        if noun not in w2i: continue
        noun_idx = w2i[noun]
        anps_to_noun_pd[anp] = noun_pd[noun_idx]
        all_nouns.add(noun_idx)

    #pre-calculate E(a | n, w, I) for all a
    #try using each of the nouns as the next word
    #all_nscore = {}
    #all_best_score = {}
    #for n in all_nouns:
    #    nscore = scores_from_noun(n)
    #    all_nscore[n] = nscore
    #    all_best_score[n] = np.amax(nscore)

    #select E(a | n, w, I) for only the a's we need
    #anps_to_adj_fluency = {}
    #for anp in anps_to_senti_score:
        #noun = anp.split('_')[1]
        #adj = anp.split('_')[0]
        #if noun not in w2i: continue
        #if adj and adj not in w2i: continue
        #if adj not in w2i:
            #anps_to_adj_fluency[anp] = all_best_score[w2i[noun]]
        #else:
            #anps_to_adj_fluency[anp] = all_nscore[w2i[noun]][w2i[adj]]
    #print anps_to_adj_fluency

    C = np.array(C)
    C /= C[3]

    #join all the scores together
    anp_to_score = {}
    for anp in anps_to_senti_score:
        vis_score = np.log(1.0 / len(anps_to_vis_score))
        pd_score = np.log(1.0 / len(w2i))
        #adj_fluency = np.log(1.0 / len(w2i))
        senti_score = np.log(anps_to_senti_score[anp])
        if anp in anps_to_vis_score: vis_score = np.log(anps_to_vis_score[anp])
        if anp in anps_to_noun_pd: pd_score = np.log(anps_to_noun_pd[anp])
        #if anp in anps_to_adj_fluency: adj_fluency = np.log(anps_to_adj_fluency[anp])
        score = C[0] * vis_score + C[1] * pd_score + C[2] * senti_score #+ C[3] * adj_fluency
        anp_to_score[anp] = score

    z = scipy.misc.logsumexp(anp_to_score.values())
    anp_to_score = sorted(anp_to_score.items(), key = lambda x: x[1], reverse=True)
    for i in xrange(len(anp_to_score)):
        anp_to_score[i] = (anp_to_score[i][0], -(anp_to_score[i][1] - z))
    best_anp_real_score = anp_to_score[0][1]
    best_anp_val = anp_to_score[0][0]
        
    if not get_all_scores:
        return best_anp_val, best_anp_real_score
    else:
        return anp_to_score

def anp_joint_score(goal, noun, img_idx, sentiscore, vscore, best_k = -1):
    anps, senti_scores = sentiscore.get_anp_score_from_noun(noun, goal)
    if not anps and best_k == -1: return ""
    if not anps: return [(0.0, "")]
    vis_scores = np.array([vscore.get_score(img_idx, anp) for anp in anps])
    vis_scores /= vis_scores.sum()
    
    joint_scores = (senti_scores**2) * (vis_scores)
    best_score = np.argmax(joint_scores)
    best_adj = anps[best_score].split('_')[0]
    if best_k == -1:
        return best_adj
    else:
        idxs = np.argsort(-joint_scores)
        best = [(joint_scores[i], anps[i].split('_')[0]) for i in idxs]
        return best


#finds the colosest word in the vocabulary using word2vec scores
class ClosestWordFinder:
    def __init__(self, w2i):
        self.nlp = spacy.load('en')
        word_vecs = []
        word_to_i = []
        for w, i in w2i.items():
            v = self.nlp.vocab[unicode(w)].vector
            word_vecs.append(v)
            word_to_i.append(i)
        word_vecs = np.array(word_vecs).T

        self.word_vecs = word_vecs
        self.word_to_i = word_to_i
        self.w2i = w2i
    
    def get_closest_word(self, chosen_word):
        if chosen_word in self.w2i:
            return self.w2i[chosen_word]
        
        v = self.nlp.vocab[unicode(chosen_word)].vector
        i = np.dot(v, self.word_vecs).argmax()
        return self.word_to_i[i]

