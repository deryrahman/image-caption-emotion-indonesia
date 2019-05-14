Requirements:
spacy (with 'en_core_web_lg', install via python -m spacy download en_core_web_lg)
theano
python2.7
pycocoevalcap (https://github.com/tylin/coco-caption) -- place in the senticap directory


=============Training================
We suggest using the pre-trained models to avoid the extensive training times.

To train the MSCOCO part of the model run (for faster training replace device=cpu with gpu, or if you have more than one gpu specifiy eg gpu0):
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=1 python train_mscoco.py

To train the Joint part of the model run:
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=1 python train_joint.py train -s pos

Replace "pos" with "neg" to train on the negative set.

===========Testing===================

To test the full senticap model run:
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=1 python train_joint.py test -s pos

Replace "pos" with "neg" to test on the negative set.

Output:
There are three different ouputs

TEXT:
eval/output_pos  -- the generated sentiment sentences one per line
eval/output_des  -- the generated descriptive sentences one per line
eval/reference%d -- the reference sentences from the senticap dataset (3 different files) on reference per line

PICKLE:
output_data/sen_att_pos_01.pik -- see code, contains everything for easy post evaluation.

STDIO:
Three lines for every input image:
1. Styled Sentence with html color annotations (indicating state of switch variable)
2. Switch variables
3. Descriptive sentence
NB: all output lines are reversed -- please read them backwards

["<font style='background-color: #FF5C33'>street</font>", "<font style='background-color: #FF5C33'>lonely</font>", 'a', 'down', 'motorcycle', 'a', 'riding', 'man', "<font style='background-color: #FF8566'>a</font>"]
[ 0.51395881  0.61651409  0.04693893  0.02415261  0.13879065  0.05620718
  0.02386183  0.24609993  0.31227863]
['street', 'city', 'a', 'on', 'motorcycle', 'a', 'riding', 'person', 'a']
