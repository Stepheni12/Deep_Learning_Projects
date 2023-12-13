# Library imports
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import pickle
import io

# File imports
from dataset import create_dataloader
from vocab import build_vocab, Vocabulary
from model import Transformer
from utils import filter_sentences, create_masks, load_tokenizers, tokenize

# Set torch seed
torch.manual_seed(9856)

# Define path variables
en_data = "train.en"
de_data = "train.de"
en_vocab_path = "en_vocab.pkl"
de_vocab_path = "de_vocab.pkl"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Define tokenizers
spacy_de, spacy_en = load_tokenizers()

# Load vocabs from current directory or build them if they don't exist
if os.path.isfile(en_vocab_path):	
	with open(en_vocab_path, 'rb') as f:
	    en_vocab = pickle.load(f)
else:
	en_vocab = build_vocab(en_data, spacy_en, threshold=2)
	
	with open(en_vocab_path, 'wb') as f:
	    pickle.dump(en_vocab, f)

	print("Total english vocabulary size: {}".format(len(en_vocab)))
	print("Saved the vocabulary wrapper to '{}'".format(en_vocab_path))

if os.path.isfile(de_vocab_path):	
	with open(de_vocab_path, 'rb') as f:
	    de_vocab = pickle.load(f)
else:
	de_vocab = build_vocab(de_data, spacy_de, threshold=2)
	
	with open(de_vocab_path, 'wb') as f:
	    pickle.dump(de_vocab, f)

	print("Total german vocabulary size: {}".format(len(de_vocab)))
	print("Saved the vocabulary wrapper to '{}'".format(de_vocab_path))

# Define paramters
heads = 8
d_model = 240 #512
hidden = 2048
max_sequence_length = 20
num_layers = 1 #6
src_vocab_size = len(en_vocab)
tgt_vocab_size = len(de_vocab)
learning_rate = 3e-4
batch_size = 2#128
epochs = 1

# Load files and preprocess
with io.open(en_data, 'r', encoding='utf-8') as file:
    en_list = file.read().split('\n')
    
with io.open(de_data, 'r', encoding='utf-8') as file:
    de_list = file.read().split('\n')

filtered_en, filtered_de = filter_sentences(en_list, de_list, max_words=max_sequence_length)

# Create data loader
data_loader = create_dataloader(filtered_en, filtered_de, spacy_en, spacy_de, en_vocab, de_vocab,
                               max_seq_length=max_sequence_length, batch_size=batch_size)

# Define model
model = Transformer(max_sequence_length=max_sequence_length,
                    src_vocab_size=src_vocab_size,
                    tgt_vocab_size=tgt_vocab_size,
                    num_layers=num_layers,
                    heads=heads,
                    d_model=d_model,
                    hidden=hidden)


# Print parameter information
parameters = list(model.parameters())
print("Parameters:",sum(p.nelement() for p in parameters))

# Initialize weights
for params in parameters:
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

# Define loss and optimizer
opt = torch.optim.Adam(parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=de_vocab.word2idx['<pad>'],
                                reduction='none')

# Prepare model for training
model.to(device)
model.train()

# Model training loop
for epoch in range(epochs):
	for i, (src, tgt, labels) in enumerate(data_loader): # Don't think we need collate_fn for labels anymore
		src = src.to(device)
		tgt = tgt.to(device)
		labels = labels.to(torch.long).to(device)

		enc_self_mask, dec_self_mask, dec_cross_mask = create_masks(src, tgt, max_sequence_length)

		enc_self_mask = enc_self_mask.to(device)
		dec_self_mask = dec_self_mask.to(device)
		dec_cross_mask = dec_cross_mask.to(device)

		logits = model(src, tgt, enc_self_mask, dec_self_mask, dec_cross_mask)
		logits = logits.view(-1,tgt_vocab_size)
		labels = labels.view(-1)
		loss = criterion(logits, labels)
		break
		#model.zero_grad()
		#loss.backward()
		#opt.step()

		if i % 25 == 0:
			continue
			#print(loss.item())
			#torch.save(decoder.state_dict(), './decoder_{}_{}.ckpt'.format(epoch, i))
			#torch.save(encoder.state_dict(), './encoder_{}_{}.ckpt'.format(epoch, i))
            
# Save the model
#torch.save(decoder.state_dict(), './decoder_final.ckpt')
#torch.save(encoder.state_dict(), './encoder_final.ckpt')