# This version of the code is stripped of anything not required to train and produce output.
# It also just generally operates differently than the notebook as more aspects of code are inside
# functions.

# The in-depth python noteboook contains much more info on data exploration, model diagnostics, and model
# testing and checks. This version is basically just all the stuff that actually works well combined 
# into one file

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Optional
import multiprocessing

######################### Pre-Processing #########################
# Load training data
df = pd.read_csv("train.csv")

# Only leave necessary characters in tweets
df['text'] = df['text'].str.replace(r'[^a-zA-Z0-9@ ]', '', regex=True)

# Convert all tweets to lowercase
df['text'] = df['text'].str.lower()

# Convert tweets with more than 2 repeating characters to just 2
df['text'] = df['text'].str.replace(r'(.)(\1+)', r'\1\1', regex=True)

# Convert all urls to 'URL'
df['text'] = df['text'].str.replace(r'http\S+', 'URL', regex=True)

# Convert all @ user mentions to 'USER'
df['text'] = df['text'].str.replace(r'@\S+', 'USER', regex=True)

# Remove duplicate tweets
df = df.drop_duplicates(subset=['text'])

# Optionally pass a list of torch tensor file names '*.pt' representing tweet embeddings you've generated
# previously and it will add them to the df. If not the function will just generate embeddings for you
def generate_or_load_embeddings(df: pd.DataFrame, file_name: str = "embeddings1.pt", embeddings_lst: Optional[List[str]] = None):
	if embeddings_lst is not None:
		ct = 1
		for file in embeddings_lst:
			load = torch.load(file).numpy()
			load = [row for row in load]
			col_name = "embeddings" + str(ct)
			df[col_name] = load
			ct += 1
		return None

	# 768 dim embedding, but really could be replaced with whatever embedding you want. Just would need 
	# to make some code adjustments
	sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

	# Generate embeddings in parallel, this really isn't super necessary I just did it like this for fun
	def embedding(texts):
	    embs = sent_model.encode(texts)
	    return embs

	n_cores = multiprocessing.cpu_count()
	batch_size = 64

	def process_batch(i):
	    batch_texts = df['text'].iloc[i:i+batch_size].tolist()
	    batch_embeddings = embedding(batch_texts)
	    return batch_embeddings

	print("Generating embeddings")    
	embeddings = Parallel(n_jobs=n_cores)(
	    delayed(process_batch)(i) for i in tqdm(range(0, len(df), batch_size)))

	flat_embeddings = [emb for batch in embeddings for emb in batch]
	df['embeddings1'] = flat_embeddings

	embedding_data = torch.tensor(np.vstack(df['embeddings1'].values), dtype=torch.float32)
	torch.save(embedding_data, file_name)

	return None

# For running val dataset during training
def validate(model, dataloader):
    model.eval()
    val_losses = []
    val_targets = []
    val_predictions = []
    
    for inputs, labels in dataloader:
        with torch.no_grad():
            outputs = model(inputs)

            # Uncomment one or the other
            # Version for averaging losses across batches
            #loss = F.binary_cross_entropy(outputs, labels)
            #v_losses.append(loss)

            # Version for getting every loss individually
            loss_val = F.binary_cross_entropy(outputs, labels, reduction='none')
            for idx in range(loss_val.shape[0]):
                val_losses.append(loss_val[idx].mean())


            val_predictions.append(outputs.numpy())
            val_targets.append(labels.numpy())

    val_avg_loss = np.array(val_losses).mean()
    # print("Val Loss:", val_avg_loss)

    # val_predictions = np.vstack(val_predictions)
    # val_targets = np.vstack(val_targets)

    # # Binary classification metrics
    # accuracy_val = metrics.accuracy_score(val_targets, (val_predictions > 0.5).astype(int))
    # print(f'Accuracy: {accuracy_val:.4f}')
    # print()
    
    return val_avg_loss

# For visualizing some of the val dataset results during training
def visualize_val(i, epochs, loss, dataloader):
        print(f'{i:3d}/{epochs:3d}: {loss.item():.6f}')
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                viz_data = inputs[:3]
                viz_labels = labels[:3]

                out_viz = model(viz_data)

                for ct in range(out_viz.shape[0]):
                    loss_viz = F.binary_cross_entropy(out_viz[ct], viz_labels[ct])
                    print("Loss:", loss_viz)
                    print("Pred:", out_viz[ct].numpy())
                    print("Pred:",  (out_viz[ct] > 0.5).float().numpy())
                    print("Targ:", viz_labels[ct].numpy())
                    print()
                break

######################### Setup for Training #########################

# Set seed for reproducibility
torch.manual_seed(4975)

# Passing the list because I already have the embeddings generated locally
generate_or_load_embeddings(df = df, embeddings_lst = ["embeddings1.pt", "embeddings2.pt"])
#generate_or_load_embeddings(df = df, file_name = "embeddings1.pt")

# Uncomment whichever embedding you intend to use
#embedding_data = torch.tensor(np.vstack(df['embeddings1'].values), dtype=torch.float32)
embedding_data = torch.tensor(np.vstack(df['embeddings2'].values), dtype=torch.float32)

#combined_data = torch.cat((keyword_data.reshape(-1,1), embedding_data), axis=1)
targets = torch.tensor(df['target'].values, dtype=torch.float32).reshape(-1,1)

# Create train, val, and test DataLoaders
dataset = TensorDataset(embedding_data, targets)

# Generate train, val, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create model
n_hidden1 = 16
n_hidden2 = 8
embedding_size = embedding_data.shape[1]

model = torch.nn.Sequential(
    torch.nn.Linear(embedding_size, n_hidden1, bias=False),
    torch.nn.BatchNorm1d(n_hidden1),
    #torch.nn.Dropout(0.4),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden1, n_hidden2, bias=False),
    torch.nn.BatchNorm1d(n_hidden2),
    #torch.nn.Dropout(0.4),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden2, 1, bias=True),
    torch.nn.Sigmoid()
)

# Implement proper weight initialization
def initialize_weights(model):
    for layer in model[:-2]:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    # Init last layer with xavier as it has a sigmoid activation instead of ReLU
    torch.nn.init.xavier_normal_(model[-2].weight, gain=1.0)

initialize_weights(model)

# Make last layer less confident (Not sure about adding this yet)
with torch.no_grad():
    model[-2].weight *= 0.1
    
######################### Model Diagnostics #########################

# This is a bunch of code to track how the network is training after every run. I was going to leave this
# out but I haven't seen much of this online so despite having it all commented out I wanted to include
# it in case people don't looking into the python notebook version that has all the graphs for this stuff
    
# Retain relu outputs for diagnostics
# last_relu_outputs = [None] * sum(1 for layer in model if isinstance(layer, torch.nn.modules.activation.ReLU))
# last_relu_grads = [None] * sum(1 for layer in model if isinstance(layer, torch.nn.modules.activation.ReLU))

# hook_handles = []

# def hook_fn_back(index):
#     def fn_b(module, grad_in, grad_out):
#         last_relu_grads[index] = grad_out[0].clone()
#     return fn_b

# layer_index = 0
# for layer in model:
#     if isinstance(layer, torch.nn.modules.activation.ReLU):
#         hook_handles.append(layer.register_full_backward_hook(hook_fn_back(layer_index)))
#         layer_index += 1

# def hook_fn(index):
#     def fn(module, inputs, outputs):
#         last_relu_outputs[index] = outputs
#     return fn

# layer_index = 0
# for layer in model:
#     if isinstance(layer, torch.nn.modules.activation.ReLU):
#         hook_handles.append(layer.register_forward_hook(hook_fn(layer_index)))
#         layer_index += 1
        
# def compute_ratio(parameter, lr):
#     gradient_std = parameter.grad.std()
#     data_std = parameter.data.std()  # Calculate data standard deviation for the same layer
#     ratio = (lr * gradient_std / data_std).log10().item()
#     return ratio
    
# Rookie Tip: I wasted a decent amount of time on this You have to wrap it in a list because 
# model.parameters() is an iterator and after you go over it once it becomes exhausted for every
# instance after.
parameters = list(model.parameters())

# Total number of parameters
print("Parameters:",sum(p.nelement() for p in parameters))
print()

for p in parameters:
    p.requires_grad = True

######################### Training #########################
model.train()

#opt = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.2)
opt = optim.Adam(model.parameters(), lr=0.0003)

epochs = 200
patience = 5 # for early stopping
best_val_loss = float('inf')
counter = 0

# For tracking metrics
lossi = []
ud = []

for i in range(epochs):
    for inputs, labels in train_loader:
        
        # forward pass
        logits = model(inputs)
        loss = F.binary_cross_entropy(logits, labels, reduction='mean')

        # backward pass
        opt.zero_grad()
        loss.backward()

        # update: simple SGD
        opt.step()
        
        # Diagnostics
        # with torch.no_grad():
        #     ud.append([compute_ratio(p, opt.param_groups[0]['lr']) for p in model.parameters()])

    lossi.append(loss.item())
    
    # Visualize val examples every 10 epochs.
    if i % 10 == 0 or i == (epochs-1): # print every once in a while
        visualize_val(i, epochs, loss, val_loader)
    
    # Validation
    val_loss = validate(model, val_loader)
    
    # Check if metric has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    
    # If no metric improvement and patience is reached, stop training loop
    if counter >= patience:
    	# Print visuals once more prior to ending training
    	visualize_val(i, epochs, loss, val_loader)
    	print(f"Early stopping after {i + 1} epochs. Best validation loss: {best_val_loss}\n")
    	break
    
#     if i >= 100:
#         break # DEBUGGING

###### Diagnostics cleanup ######
# for handle in hook_handles:
#     handle.remove()

# Save model for deployment
model_scripted = torch.jit.script(model)
model_scripted.save('model.pth')

######################### Post-Training Evaluation #########################

model.eval()
t_predictions = []
t_targets = []
v_predictions = []
v_targets = []

threshold = 0.5

# Train Loss Calculations
with torch.no_grad():
    t_losses = []
    for inputs, labels in train_loader:
        outputs = model(inputs)

        # Version for averaging losses across batches
        # loss = F.binary_cross_entropy(outputs, labels)
        # t_losses.append(loss)

        # Version for getting every loss individually
        loss = F.binary_cross_entropy(outputs, labels, reduction='none')
        for idx in range(loss.shape[0]):
            t_losses.append(loss[idx].mean())

        t_predictions.append(outputs.numpy())
        t_targets.append(labels.numpy())
        
train_avg_loss = np.array(t_losses).mean()
print("Train Loss:", train_avg_loss)

t_predictions = np.vstack(t_predictions)
t_targets = np.vstack(t_targets)

# Binary classification metrics
train_accuracy = metrics.accuracy_score(t_targets, (t_predictions > threshold).astype(int))
train_f1 = metrics.f1_score(t_targets, (t_predictions > threshold).astype(int))

print(f'Accuracy: {train_accuracy:.4f}')
print(f'F1-score: {train_f1:.4f}\n')

# Validation Loss Calculations
with torch.no_grad():
    v_losses = []
    for inputs, labels in val_loader:
        outputs = model(inputs)

        # Version for averaging losses across batches
        #loss = F.binary_cross_entropy(outputs, labels)
        #v_losses.append(loss)

        # Version for getting every loss individually
        loss = F.binary_cross_entropy(outputs, labels, reduction='none')
        for idx in range(loss.shape[0]):
            v_losses.append(loss[idx].mean())

        v_predictions.append(outputs.numpy())
        v_targets.append(labels.numpy())
        
val_avg_loss = np.array(v_losses).mean()
print("Val Loss:", val_avg_loss)

v_predictions = np.vstack(v_predictions)
v_targets = np.vstack(v_targets)

# Binary classification metrics
accuracy = metrics.accuracy_score(v_targets, (v_predictions > threshold).astype(int))
f1 = metrics.f1_score(v_targets, (v_predictions > threshold).astype(int))

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}\n')

######################### Test Set Evaluation #########################
# Uncomment section below when you wish to evaluate against the test set we generated previously

test_predictions = []
test_targets = []
threshold = 0.5

# Test Loss Calculations
with torch.no_grad():
    test_losses = []
    for inputs, labels in test_loader:
        outputs = model(inputs)

        # Version for averaging losses across batches
        #loss = F.binary_cross_entropy(outputs, labels)
        #test_losses.append(loss)

        # Version for getting every loss individually
        loss = F.binary_cross_entropy(outputs, labels, reduction='none')
        for idx in range(loss.shape[0]):
            test_losses.append(loss[idx].mean())

        test_predictions.append(outputs.numpy())
        test_targets.append(labels.numpy())
        
test_avg_loss = np.array(test_losses).mean()
print("Test Loss:", test_avg_loss)

test_predictions = np.vstack(test_predictions)
test_targets = np.vstack(test_targets)

# Binary classification metrics
accuracy = metrics.accuracy_score(test_targets, (test_predictions > threshold).astype(int))
f1 = metrics.f1_score(v_targets, (v_predictions > threshold).astype(int))

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}\n')

######################### Kaggle Test Set Evaluation #########################

# This was part of a kaggle competition so this preps the kaggle test set and runs it through the model
test_df = pd.read_csv('test.csv')

# Pre-Processing
# Only leave necessary characters in tweets
test_df['text'] = test_df['text'].str.replace(r'[^a-zA-Z0-9@ ]', '', regex=True)

# Convert all tweets to lowercase
test_df['text'] = test_df['text'].str.lower()

# Convert tweets with more than 2 repeating characters to just 2
test_df['text'] = test_df['text'].str.replace(r'(.)(\1+)', r'\1\1', regex=True)

# Convert all urls to 'URL'
test_df['text'] = test_df['text'].str.replace(r'http\S+', 'URL', regex=True)

# Convert all @ user mentions to 'USER'
test_df['text'] = test_df['text'].str.replace(r'@\S+', 'USER', regex=True)

#generate_or_load_embeddings(df = test_df, embeddings_lst = ["test_embeddings1.pt", "test_embeddings2.pt"])
generate_or_load_embeddings(df = test_df, file_name = "test_embeddings1.pt")

def model_test(emb):
    model.eval()
    emb = torch.from_numpy(emb)
    with torch.no_grad():
        out = model(emb.reshape(1,-1))
        return (out > 0.5).int().item()

# For the submission the column name needs to be 'target'
#test_df['target'] = test_df['embeddings1'].apply(model_test)
test_df['target'] = test_df['embeddings1'].apply(model_test)


submission_df = test_df[['id', 'target']]
submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"Kaggle test submission saved to {submission_file}\n")