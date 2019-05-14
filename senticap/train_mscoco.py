from mrnn.mrnn import *
from mrnn.mrnn_algorithms import *

conf = {
    'VAL_SPLIT': RNNDataProvider.TEST_VAL,
    'BATCH_NORM': False,
    'SEMI_FORCED': 1,
    'REVERSE_TEXT': True,
    'DROP_MASK_SIZE_DIV': 16,
    'batch_size_val': 512,
    'DATASET': RNNDataProvider.COCO,
    'emb_size': 512,
    'lstm_hidden_size': 512
}
rnn = RNNModel(conf)
#rnn.load_model("saved_models/saved_model_mscoco_110.pik")
rnn.conf['VAL_SPLIT'] = RNNDataProvider.VAL
rnn.conf['TRAIN_SPLIT'] = RNNDataProvider.TRAIN
rnn.load_training_dataset()
rnn.build_model_core()
rnn.load_val_dataset()

rnn.build_sentence_generator()

rnn.build_perplexity_calculator()
#print rnn.get_val_perplexity()
#for i in xrange(10):
#    idx = np.random.randint(rnn.V_valid.shape[0])
#    print decoder_beamsearch(rnn, rnn.V_valid[idx])#np.zeros((1,)))
#sys.exit(0)

#for i in xrange(10):
#idx = np.random.randint(rnn.V_valid.shape[0])
#    idx = i
#    print rnn.Id_valid[idx]
#    print rnn.get_sentence(rnn.V_valid[idx])#np.zeros((1,)))
#sys.exit(0)

#print "PPL: %f" % rnn.get_val_perplexity()
#sys.exit(0)

rnn.build_model_trainer()


def iter_callback(rnn, num_epoch):
    print num_epoch
    for i in xrange(10):
        idx = np.random.randint(rnn.V_valid.shape[0])
        print rnn.get_sentence(rnn.V_valid[idx])  #np.zeros((1,)))


def epoch_callback(rnn, num_epoch):
    rnn.save_model("saved_models/saved_model_mscoco_new_%d.pik" % num_epoch)
    print "PPL: %f" % rnn.get_val_perplexity()


rnn.train_complete(iter_cb_freq=10,
                   iter_callback=iter_callback,
                   epoch_callback=epoch_callback)
