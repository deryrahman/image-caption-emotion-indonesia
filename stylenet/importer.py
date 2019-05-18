import torch
import pickle
from build_vocab import Vocabulary
from model_att import EncoderCNN, DecoderFactoredLSTMAtt
from utils import save_checkpoint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = '../data/flickr8k_id/vocab_051619.pkl'
checkpoint_path = './models/checkpoint_stylenet_att.pth.tar'

checkpoint = torch.load(checkpoint_path)
start_epoch = checkpoint['epoch'] + 1
epochs_since_improvement = checkpoint['epochs_since_improvement']
best_bleu4 = checkpoint['bleu-4']
decoder = checkpoint['decoder']
encoder = checkpoint['encoder']
optimizer = checkpoint['optimizer']
lang_optimizer = checkpoint['lang_optimizer']

# Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Build models
encoder = encoder.to(device)
decoder = decoder.to(device)
decoder2 = DecoderFactoredLSTMAtt(512, 300, 512, 512, len(vocab), 1).to(device)

# print(encoder)
print(list(decoder.f_beta.parameters()))
print(list(decoder2.f_beta.parameters()))
