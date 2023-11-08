import pandas as pd
import os
import nltk
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from vocab import Vocabulary

# Create dataset class
class FlickrDataset(Dataset):
    """Flickr Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_dir, dataframe, vocab, image_processor):
        
        self.image_dir = image_dir              # Directory storing images
        self.dataframe = dataframe              # Dataframe containing image file names and corresponding caption
        self.vocab = vocab                      # Vocabulary wrapper built from the dataframe captions
        self.image_processor = image_processor  # Image processor that preprocesses images for the hugging face model
    
    def __getitem__(self, idx):
        vocab = self.vocab
        
        row = self.dataframe.iloc[idx]
        image_file = row.iloc[0]
        caption = row.iloc[1]
        
        tokens = nltk.tokenize.word_tokenize(caption)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.tensor(caption, dtype=torch.long)
        
        raw_image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
        processed_image = self.image_processor(images=raw_image, return_tensors="pt")
        image = processed_image.pixel_values.squeeze(0)
        
        return image, target
    
    def viewImage(self, idx):
    
        row = self.dataframe.iloc[idx]
        image_file = row.iloc[0]
        caption = row.iloc[1]
        image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            
        return image, caption
    
    def __len__(self):
        return len(self.dataframe)


def collate_fn(data):
    """Custom collate function to handle more complex training batch and prepare for packing."""

    # Sort the batch data by caption length (descending order)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Stack images to produce 4D tensor to input to the model
    images = torch.stack(images, 0)

    # Pad captions to produce 2D tensor to input to the model
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths


def create_dataloader(image_dir, dataframe, vocab, image_processor, batch_size=128):
    """Creates dataloader for kaggle Flickr8K dataset."""

    flickr = FlickrDataset(image_dir, dataframe, vocab, image_processor)

    # This will return (images, captions, lengths) for each iteration
    # images: a tensor of shape (batch_size, 3, 256, 256)
    # captions: a tensor of shape (batch_size, padded_length) 
    #   - Where padded_length will change depending on the captions in the current batch
    # lengths: a list indicating the true length of each caption prior to padding. 
    #   - The length of this list is equivalent to the batch size.
    data_loader = torch.utils.data.DataLoader(dataset=flickr, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              collate_fn=collate_fn)
    
    return data_loader