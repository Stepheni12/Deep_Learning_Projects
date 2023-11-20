import torch
from torch.utils.data import Dataset, DataLoader

from vocab import Vocabulary
from utils import tokenize

# Create dataset class
class Multi30k(Dataset):
    
    def __init__(self, en_list, de_list, en_tokenizer, de_tokenizer, en_vocab, de_vocab, max_seq_len):
        
        self.en_list = en_list
        self.de_list = de_list
        self.en_tokenizer = en_tokenizer
        self.de_tokenizer = de_tokenizer
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, idx):
        
        en_sent = self.en_list[idx]
        de_sent = self.de_list[idx]
        
        en_tok = tokenize(en_sent, self.en_tokenizer)
        de_tok = tokenize(de_sent, self.de_tokenizer)
        
        en_vect = []
        de_vect = []
        
        en_vect.append(self.en_vocab('<start>'))
        de_vect.append(self.de_vocab('<start>'))
        en_vect.extend([self.en_vocab(token) for token in en_tok])
        de_vect.extend([self.de_vocab(token) for token in de_tok])
        
        en_vect.append(self.en_vocab('<end>'))
        de_vect.append(self.de_vocab('<end>'))
        
        max_seq = self.max_seq_len
            
        if len(en_vect) < max_seq:
            tmp = [0] * (max_seq - len(en_vect))
            en_vect.extend(tmp)
            
        if len(de_vect) < max_seq:
            tmp = [0] * (max_seq - len(de_vect))
            de_vect.extend(tmp)
        
        src = torch.tensor(en_vect, dtype=torch.long)
        tgt = torch.tensor(de_vect, dtype=torch.long)
        
        return src, tgt
    
    def viewSentences(self, idx):
    
        en = self.en_list[idx]
        de = self.de_list[idx]
            
        return en, de
    
    def __len__(self):
        return len(self.en_list)


def collate_fn(data):
    
    src, tgt = zip(*data)
    
    src = torch.stack(src, 0)
    tgt = torch.stack(tgt, 0)

    # Label is made from removing start token and replacing with padding token at the end (padding token=0)
    labels = tgt[:, 1:]
    append_pad = torch.zeros(tgt.shape[0],1)
    labels = torch.cat((labels, append_pad), dim=1)
     
    return src, tgt, labels


def create_dataloader(en_list, de_list, en_tokenizer, de_tokenizer, en_vocab, de_vocab, max_seq_length, batch_size):
    data = Multi30k(en_list, de_list, en_tokenizer, de_tokenizer, en_vocab, de_vocab, max_seq_length)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    return data_loader