from sentiment_utils import *
from nltk.stem import WordNetLemmatizer

from mrnn.mrnn_switched import *
from mrnn.mrnn_algorithms import *
import mrnn.mrnn_io
import cPickle
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import spacy
import argparse

nlp = spacy.load('en_core_web_lg')

wnl = WordNetLemmatizer()
sentiscore = SentiScores("./data/noun_to_adj_senti_mine+dsb_0.100.pik")
anp_list = set()
for noun in sentiscore.noun_to_senti:
    for adj, s in sentiscore.noun_to_senti[noun]:
        anp_list.add((wnl.lemmatize(adj), wnl.lemmatize(noun)))


def sentence_has_anp(sentence):
    words = sentence
    found = False
    for wi in xrange(len(words) - 1):
        w1 = wnl.lemmatize(words[wi])
        w2 = wnl.lemmatize(words[wi + 1])
        if (w1, w2) in anp_list:
            return True
    return False


def calculate_metric(rnn, meteor=None):
    gts = {}
    res = {}
    lp_avg = 0.0
    lp_c = 0
    for idx in xrange(rnn.V_valid.shape[0]):
        iid = rnn.Id_valid[idx]
        if iid not in gts: gts[iid] = []
        #gts[iid].append(' '.join([rnn.dp.i2w[w] for w in rnn.X_valid[idx] if w != 0][::-1]))
        gts[iid] = [
            ' '.join(rnn.dp.tokens[i][::-1])
            for i in rnn.dp.img_id_to_tokens[iid]
        ]
        if iid in res: continue
        res[iid] = []
        #pos_sen, pos_att = rnn.get_sentence(rnn.V_valid[idx], senti=np.array([1.0], dtype=theano.config.floatX))
        (lp, pos_sen) = decoder_beamsearch(rnn,
                                           rnn.V_valid[idx],
                                           senti=1.0,
                                           beam_size=1)
        pos_sen = pos_sen[:-1]
        print ' '.join(pos_sen[::-1])
        res[iid].append(' '.join(pos_sen[::-1]))
        lp_avg += np.exp(lp)
        lp_c += 1
    lp_avg /= float(lp_c)
    return lp_avg

    bleu = Bleu()
    print "Bleu:"
    print "Positive:", bleu.compute_score(gts, res)[0]
    rouge = Rouge()
    print "Rouge:"
    print "Positive:", rouge.compute_score(gts, res)[0]
    if meteor is None:
        meteor = Meteor()
    print "Meteor:"
    mscore = meteor.compute_score(gts, res)[0]
    print "Positive:", mscore
    return mscore


def do_replace_adj(sentence):
    tokens = nlp(unicode(' '.join(sentence)))
    out_tokens = []
    for t in tokens:
        if t.pos == spacy.parts_of_speech.ADJ:
            out_tokens.append("NOTAWORDATALL")
        else:
            out_tokens.append(str(t.orth_))
    return out_tokens


def run_load_gap_filler(pretrained_filename,
                        do_bleu=False,
                        must_have_anp=False,
                        copy_if_no_anp=False,
                        replace_adj=False,
                        get_human=False,
                        semi_human=False):
    rnn = RNNModel()
    rnn.load_model(pretrained_filename)
    rnn.conf['VAL_SPLIT'] = RNNDataProvider.TEST

    if get_human:
        id_to_caps = cPickle.load(open("coco_mturk/id_to_caps.pik", "rb"))

    rnn.build_model_core()
    rnn.load_val_dataset()

    rnn.build_sentence_generator()

    rnn.build_perplexity_calculator()
    #print rnn.sample_sentence(rnn.V_valid[0])
    #print decoder_beamsearch2(rnn, rnn.V_valid[0])
    #print decoder_beamsearch(rnn, rnn.V_valid[0])

    #calculate_metric(rnn)
    #sys.exit(0)

    pos_sentence_res = []
    pos_att_res = []

    des_sentence_res = []
    des_att_res = []

    img_files = []
    img_ids = []

    id_to_sentences = {}

    seen_ids = set()
    if 'added_words' in rnn.conf:
        new_words = set([w[0] for w in rnn.conf['added_words']])
    else:
        new_words = set()
    num_ignore = 0
    num_not_ignore = 0
    for idx in xrange(rnn.V_valid.shape[0]):
        img_file = rnn.dp.img_id_to_filename[rnn.Id_valid[idx]]
        img_id = rnn.Id_valid[idx]
        if img_id not in id_to_sentences: id_to_sentences[img_id] = []
        #id_to_sentences[img_id].append(' '.join([rnn.dp.i2w[w] for w in rnn.X_valid[idx] if w != 0][::-1]))
        if replace_adj:
            id_to_sentences[img_id] = [
                ' '.join(do_replace_adj(rnn.dp.tokens[i])[::-1])
                for i in rnn.dp.img_id_to_tokens[img_id]
            ]
        elif get_human:
            id_to_sentences[img_id] = [
                ' '.join(rnn.dp.tokens[i][::-1])
                for i in rnn.dp.img_id_to_tokens[img_id]
            ]
            np.random.shuffle(id_to_sentences[img_id])
            print len(id_to_sentences[img_id])
            human_sen_pos = id_to_sentences[img_id].pop()
            print len(id_to_sentences[img_id])
            if not id_to_sentences[img_id]: continue
        else:
            id_to_sentences[img_id] = [
                ' '.join(rnn.dp.tokens[i][::-1])
                for i in rnn.dp.img_id_to_tokens[img_id]
            ]
        #print id_to_sentences[img_id]
        if img_id in seen_ids: continue
        seen_ids.add(img_id)
        if get_human and not semi_human:
            pos_sen = human_sen_pos.split()[::-1]
            np.random.shuffle(id_to_caps[img_id])
            des_sen = id_to_caps[img_id][0][::-1]
        else:
            lp, pos_sen, pos_att = decoder_beamsearch_with_attention(
                rnn, rnn.V_valid[idx], senti=1.0, beam_size=5)
            lp, des_sen, des_att = decoder_beamsearch_with_attention(
                rnn, rnn.V_valid[idx], senti=-1.0, beam_size=5)
            pos_sen = pos_sen[:-1]
            des_sen = des_sen[:-1]
            #des_att = des_att[:-1]
            pos_att = pos_att[:-1]
        #pos_sen, pos_att = rnn.get_sentence(rnn.V_valid[idx], senti=np.array([1.0], dtype=theano.config.floatX))
        pos_att = np.array(pos_att)
        pos_att = pos_att.flatten()
        #des_att = np.array(des_att)
        #des_att = des_att.flatten()
        des_att = np.zeros((len(des_sen),))
        #pos_att = np.zeros((len(pos_sen),))
        if must_have_anp:
            if not sentence_has_anp(pos_sen[::-1]):
                num_ignore += 1
                continue
            num_not_ignore += 1
        if copy_if_no_anp:
            if not sentence_has_anp(pos_sen[::-1]):
                pos_sen = des_sen
        if replace_adj:
            pos_sen = do_replace_adj(pos_sen[::-1])[::-1]
            des_sen = do_replace_adj(des_sen[::-1])[::-1]

        #des_sen, des_att = rnn.get_sentence(rnn.V_valid[idx], senti=np.array([-1.0], dtype=theano.config.floatX))
        new_pos_sen = []
        for vv, a in zip(pos_sen, pos_att):
            out = vv
            col = ""
            if a > 0.75:
                col = "#FF3300"
            elif a > 0.5:
                col = "#FF5C33"
            elif a > 0.25:
                col = "#FF8566"
            #if a > 0.75:
            #    col = "#33CC33"# "#3366FF"
            #elif a > 0.5:
            #    col = "#70DB70" #"#5C85FF"
            #elif a > 0.25:
            #    col = "#ADEBAD" #"#85A3FF"
            if col:
                out = "<font style='background-color: %s'>%s</font>" % (col, vv)
            new_pos_sen.append(out)
        pos_sen = new_pos_sen
        print pos_sen
        print pos_att
        print des_sen
        print_it = False
        for v in pos_sen:
            if v in new_words:
                print_it = True
        if print_it:
            for x in zip(pos_sen, pos_att)[::-1]:
                print x[0],
            print ""
        #for x in zip(pos_sen, pos_att)[::-1]:
        #    print x[0],
        #print ""
        #for x in zip(des_sen, des_att)[::-1]:
        #    print x[0],
        #print "\n"
        pos_att = pos_att[:len(pos_sen)]
        des_att = des_att[:len(des_sen)]
        pos_sentence_res.append(pos_sen[::-1])
        pos_att_res.append(np.exp(pos_att[::-1]))
        des_sentence_res.append(des_sen[::-1])
        des_att_res.append(np.exp(des_att[::-1]))
        img_files.append(img_file)
        img_ids.append(img_id)

    output = {
        'pos_sen': pos_sentence_res,
        'pos_att': pos_att_res,
        'des_sen': des_sentence_res,
        'des_att': des_att_res,
        'img_files': img_files,
        'img_ids': img_ids
    }
    cPickle.dump(output,
                 open("output_data/sen_att_pos_01.pik", "wb"),
                 protocol=2)

    if must_have_anp:
        print "Must have ANP % removed:", num_ignore / float(
            num_not_ignore) * 100.0

    print "getting Positive perplexity"
    print rnn.get_val_perplexity()
    print "got perplexity"

    print "getting Descriptive perplexity"
    print rnn.get_val_perplexity(base=True)
    print "got perplexity"

    gts = {}
    res = {}
    fout = open("eval/output_pos", "w")
    for line, iid in zip(pos_sentence_res, img_ids):
        fout.write(' '.join(line) + '\n')
        if iid not in res: res[iid] = []
        res[iid].append(' '.join(line))
    fout.close()

    res_des = {}
    fout = open("eval/output_des", "w")
    for line, iid in zip(des_sentence_res, img_ids):
        fout.write(' '.join(line) + '\n')
        if iid not in res_des: res_des[iid] = []
        res_des[iid].append(' '.join(line))
    fout.close()

    for i in xrange(3):
        fout = open("eval/reference%d" % i, "w")
        for cid in img_ids:
            if cid not in gts: gts[cid] = []
            if len(id_to_sentences[cid]) > i:
                gts[cid].append(id_to_sentences[cid][i])
                fout.write(id_to_sentences[cid][i] + "\n")
            else:
                fout.write("\n")
        fout.close()

    bleu = Bleu()
    #for i in gts.keys()[:10]:
    #    print gts[i]
    #    print res_des[i]
    #    print res[i]
    #    print ""
    total_ref_sentences = 0
    for i in gts.keys():
        total_ref_sentences += len(gts[i])
    print "Total ref sentences:", total_ref_sentences
    print "Bleu:"
    print "Positive:", bleu.compute_score(gts, res)[0]
    print "Descriptive:", bleu.compute_score(gts, res_des)[0]
    rouge = Rouge()
    print "Rouge:"
    print "Positive:", rouge.compute_score(gts, res)[0]
    print "Descriptive:", rouge.compute_score(gts, res_des)[0]
    cider = Cider()
    print "Cider:"
    print "Positive:", cider.compute_score(gts, res)[0]
    print "Descriptive:", cider.compute_score(gts, res_des)[0]
    meteor = Meteor()
    print "Meteor:"
    print "Positive:", meteor.compute_score(gts, res)[0]
    print "Descriptive:", meteor.compute_score(gts, res_des)[0]


def run_train_gap_filler(lambda_n=0.25,
                         lambda_gam=0.25,
                         fixed_alpha=0.3,
                         similar_param_reg=10,
                         domain_adapt=DA_SUM,
                         style_mode="pos"):
    conf = {
        'VAL_SPLIT': RNNDataProvider.VAL,
        'BATCH_NORM': False,
        'SEMI_FORCED': 1,
        'REVERSE_TEXT': True,
        'DROP_MASK_SIZE_DIV': 16,
        'batch_size_val': 128,
        'DATASET': RNNDataProvider.COCO_MTURK,
        'emb_size': 512,
        'lstm_hidden_size': 512
    }

    #load the existing model
    rnn = RNNModel(conf)
    saved_model_filename = "saved_models/saved_model_mscoco_110.pik"
    if domain_adapt == DA_SIMILAR_PARAM_SEPARATE:
        saved_model_filename = "saved_models/saved_model_mscoco_gap_filler_pos2_fxA_similar_3.pik"
    rnn.load_model(saved_model_filename,
                   conf={
                       'DATASET': RNNDataProvider.COCO_MTURK,
                       'VAL_SPLIT': RNNDataProvider.VAL,
                       'TRAIN_SPLIT': RNNDataProvider.TRAIN,
                       'batch_size_val': 128,
                       'SWITCHED': True,
                   })
    rnn.conf['VAL_SPLIT'] = RNNDataProvider.VAL
    rnn.conf['param_names_trainable'] = [
        "wemb_sw", "w_sw", "b_sw", "w_lstm_sw", "att_w", "att_b", "wsenti",
        "wvm_sw", "bmv_sw"
    ]
    rnn.conf['param_names_saveable'] = rnn.conf['param_names_saveable'] + [
        "wemb_sw", "w_sw", "b_sw", "w_lstm_sw", "att_w", "att_b", "wsenti",
        "wvm_sw", "bmv_sw"
    ]
    #rnn.conf['LEARNING_RATE'] = 0.0001
    rnn.conf['LAMBDA_N'] = lambda_n
    rnn.conf['LAMBDA_GAM'] = lambda_gam
    rnn.conf['DATASET'] = RNNDataProvider.COCO_MTURK
    rnn.conf['DOMAIN_ADAPT'] = domain_adapt
    if domain_adapt == DA_SIMILAR_PARAM_SEPARATE:
        rnn.conf['DOMAIN_ADAPT'] = DA_SUM
    rnn.conf['FIXED_ALPHA'] = fixed_alpha
    rnn.conf['SIMILAR_PARAM_REG'] = similar_param_reg
    #rnn.conf['GRAD_METHOD'] = ADADELTA
    #rnn.conf['param_names_trainable'] = ["wemb_sw", "w_sw", "b_sw", "w_lstm_sw", "wemb", "wvm", "bmv", "w", "b", "w_lstm", "att_w", "att_b"]
    rnn.load_training_dataset(merge_vocab=True)
    rnn.build_model_core()
    rnn.load_val_dataset()

    rnn.build_sentence_generator()

    rnn.build_perplexity_calculator()
    print "getting perplexity"
    print rnn.get_val_perplexity()
    print "got perplexity"
    #sys.exit(0)

    rnn.build_model_trainer()

    def iter_callback(rnn, num_epoch):
        print num_epoch
        return
        for i in xrange(10):
            idx = np.random.randint(rnn.V_valid.shape[0])
            print "Orig:",
            sen, att = rnn.get_sentence(rnn.V_valid[idx],
                                        senti=np.array(
                                            [-1.0], dtype=theano.config.floatX))
            for x in zip(sen, att)[::-1]:
                print x[0],
            print ""
            print "Sen+:",
            sen, att = rnn.get_sentence(rnn.V_valid[idx],
                                        senti=np.array(
                                            [1.0], dtype=theano.config.floatX))
            for x in zip(sen, att)[::-1]:
                print x[0],
            print ""
            print "Sen+:",
            sen, att = rnn.get_sentence(rnn.V_valid[idx],
                                        senti=np.array(
                                            [1.0], dtype=theano.config.floatX))
            for x in zip(sen, att)[::-1]:
                print "%s_%.2f" % (x[0], np.exp(x[1])),
            print ""
            #print "Sen-:",
            #sen, att = rnn.get_sentence(rnn.V_valid[idx], senti=np.array([0.0], dtype=theano.config.floatX))
            #for x in zip(sen, att)[::-1]:
            #print "(%s, %.3f)" % (x[0], x[1]),
            #    print x[0],
            print "\n"

    best_ppl = 1000
    fail_count = 15
    epoch_args = {}
    epoch_args['best_ppl'] = best_ppl
    epoch_args['fail_count'] = fail_count
    epoch_args['meteor'] = Meteor()

    def epoch_callback(rnn, num_epoch, epoch_args):
        ppl = rnn.get_val_perplexity()
        #ppl = -calculate_metric(rnn, meteor = epoch_args['meteor'])
        print "PPL: %f" % ppl
        if ppl < epoch_args['best_ppl']:
            epoch_args['fail_count'] = 15
            epoch_args['best_ppl'] = ppl
            rnn.save_model(
                "saved_models/saved_model_mscoco_gap_filler_pos2_fxA_%s_model0+model1r_new.pik"
                % style_mode)
        else:
            #rnn.save_model("saved_models/saved_model_mscoco_gap_filler_pos_%d_bppl.pik" % num_epoch)
            epoch_args['fail_count'] -= 1

        if epoch_args['fail_count'] == 0:
            return epoch_args['best_ppl']

    print rnn.sample_sentence(rnn.V_valid[0])
    best_ppl = rnn.train_complete(iter_cb_freq=5,
                                  iter_callback=iter_callback,
                                  epoch_callback=epoch_callback,
                                  epoch_args=epoch_args,
                                  epoch_at_iter=True)
    return best_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_mode", choices=["test", "train"])
    ap.add_argument("-s",
                    "--style_mode",
                    choices=["pos", "neg"],
                    default="pos",
                    help="Generate or train on positive or negative style")
    args = ap.parse_args()
    if args.run_mode == "test":
        if args.style_mode == "pos":
            pretrained_filename = "saved_models/saved_model_mscoco_gap_filler_pos2_fxA_pos_model0+model1r.pik"
        else:
            pretrained_filename = "saved_models/saved_model_mscoco_gap_filler_pos2_fxA_neg_model0+model1r.pik"
        run_load_gap_filler(pretrained_filename)
    elif args.run_mode == "train":
        if args.style_mode == "pos":
            mrnn.mrnn_io.DO_NEG = False
        else:
            mrnn.mrnn_io.DO_NEG = True
        ppl_final = run_train_gap_filler(lambda_n=0.0697021,
                                         lambda_gam=np.exp(-0.640564),
                                         domain_adapt=DA_SIMILAR_PARAM_2,
                                         similar_param_reg=3**0,
                                         style_mode=args.style_mode)
        print "PPL Final:" + ppl_final

        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564))
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564), domain_adapt=DA_FIXED_ALPHA, fixed_alpha=0.45)
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564), domain_adapt=DA_FIXED_ALPHA, fixed_alpha=1)
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564), domain_adapt=DA_SIMILAR_PARAM, similar_param_reg=3**0)
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564), domain_adapt=DA_SIMILAR_PARAM_3, similar_param_reg=3**0)
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0901334, lambda_gam=np.exp(-0.164476), domain_adapt=DA_SIMILAR_PARAM_3, similar_param_reg=3**-1.93560)
        #print "PPL Final:", run_train_gap_filler(lambda_n=0.0697021, lambda_gam=np.exp(-0.640564), domain_adapt=DA_SIMILAR_PARAM_SEPARATE, similar_param_reg=3**0)
    #run_train_gap_filler(lambda_n=0.000366211, lambda_gam=0.533325)


if __name__ == "__main__":
    main()
