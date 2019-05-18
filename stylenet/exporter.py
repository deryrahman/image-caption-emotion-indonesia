import torch
import pickle
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderFactoredLSTM
from utils import save_checkpoint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = '../data/flickr8k_id/vocab_051619_30k_4.pkl'
encoder_path = './models/models_finetune_30k_4/encoder-8.ckpt'
decoder_path = './models/models_finetune_30k_4/decoder-8.ckpt'
lr_caption = 0.0002
lr_language = 0.0005
is_best = True

# start_epoch = 22 + 1
epoch = 8
epochs_since_improvement = {'factual': 0, 'emotion': 0}
bleu4 = {'factual': 0.03470848014924343, 'emotion': 0.}
# Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Build models
encoder = EncoderCNN(512).to(device)
decoder = DecoderFactoredLSTM(300, 512, 512, len(vocab), 1).to(device)

# Load the trained model parameters
# encoder.load_state_dict(torch.load(encoder_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

params = list(decoder.parameters()) + list(encoder.adaptive_pool.parameters())
lang_params = list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=lr_caption)
lang_optimizer = torch.optim.Adam(lang_params, lr=lr_language)

save_checkpoint('models', 'stylenet_finetune_30k_4', epoch,
                epochs_since_improvement, encoder, decoder, optimizer,
                lang_optimizer, bleu4, is_best)
