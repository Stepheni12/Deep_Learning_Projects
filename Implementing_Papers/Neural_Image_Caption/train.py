# Library imports
import pandas as pd
import torch
import os
import pickle
from transformers import MobileViTImageProcessor

# File imports
from data_loader import create_dataloader
from vocab import build_vocab
from vocab import Vocabulary
from model import ImageEncoderCNN, CaptionDecoderRNN

# Set torch seed
torch.manual_seed(9856)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Load csv and preprocess
df = pd.read_csv("captions.txt")
df['caption'] = df['caption'].str.lower()
df['caption'] = df['caption'].str.replace(r"[^a-zA-Z0-9-' ]", '', regex=True)

# Load vocab from current directory ("./vocab.pkl") or build vocab if it doesn't exist
vocab_path = "./vocab.pkl"
if os.path.isfile(vocab_path):	
	with open(vocab_path, 'rb') as f:
	    vocab = pickle.load(f)
else:
	vocab = build_vocab(df, threshold=4)
	
	with open(vocab_path, 'wb') as f:
	    pickle.dump(vocab, f)
	print("Total vocabulary size: {}".format(len(vocab)))
	print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

# Load embedding tensor from current directory ("./embeddings_table.pt") or build embedding table if it doesn't exist
embeddings_table_path = "./embeddings_table.pt"
if os.path.isfile(embeddings_table_path):
	embeddings_table = torch.load(embeddings_table_path)
else:
	# Load hugging face sentence transformer
	sent_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-V2')
	embeddings_lst = []

	def embedding(texts):
	    embs = sent_model.encode(texts)
	    return embs

	for i in range(len(vocab)):
	    word = vocab.idx2word[i]
	    embeddings_lst.append(embedding(word))
	    if i % 500 == 0:
	        print(i)
	        
	np_arr = np.array(embeddings_lst)
	embeddings_table = torch.Tensor(np_arr)
	torch.save(embeddings_table, embeddings_table_path)
	print("Embedding table size:", embedding_table.shape)
	print("Saved the embedding table to '{}'".format(vocab_path))

# Define image processor
ImageProcessor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-xx-small")

# Create data loader
data_loader = create_dataloader("./Images", df, vocab, ImageProcessor, batch_size=128)

# Define models
encoder = ImageEncoderCNN(embed_size=384).to(device)
decoder = CaptionDecoderRNN(embeddings_table, 512, embeddings_table.shape[1], len(vocab)).to(device)

# Set models to train mode
encoder.train()
decoder.train()

# Parameter information
all_params = list(decoder.parameters()) + list(encoder.linear.parameters())
print("Parameters:",sum(p.nelement() for p in all_params))

# Define model parameters
opt = torch.optim.Adam(all_params, lr=0.0003)
epochs = 1

# Model training loop
for epoch in range(epochs):
	for i, (images, captions, lengths) in enumerate(data_loader):
		images = images.to(device)
		captions = captions.to(device)
		targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]

		image_features = encoder(images)
		logits = decoder(image_features, captions, lengths)
		loss = torch.nn.functional.cross_entropy(logits, targets)
		decoder.zero_grad()
		encoder.zero_grad()
		loss.backward()
		opt.step()

		if i % 25 == 0:
			print("Epoch: {}, Batch: {}: {}".format(epoch, i, loss.item()))
			torch.save(decoder.state_dict(), './cpoints/decoder_{}_{}.ckpt'.format(epoch, i))
			torch.save(encoder.state_dict(), './cpoints/encoder_{}_{}.ckpt'.format(epoch, i))

# Save the model
torch.save(decoder.state_dict(), './cpoints/decoder_final.ckpt')
torch.save(encoder.state_dict(), './cpoints/encoder_final.ckpt')