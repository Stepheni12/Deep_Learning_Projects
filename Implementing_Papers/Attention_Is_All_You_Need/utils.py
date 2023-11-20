import torch
import re
import numpy as np
import spacy
import os

def filter_sentences(english_sentences, german_sentences, max_words):
    filtered_english = []
    filtered_german = []
    sum_ = 0

    for eng_sent, ger_sent in zip(english_sentences, german_sentences):
        eng_words = len(eng_sent.split())
        ger_words = len(ger_sent.split())
        
        # Subtracting two accounts for the start and stop tokens
        if eng_words <= max_words-2 and ger_words <= max_words-2:
            filtered_english.append(re.sub(r'[^\w\s]', '', eng_sent))
            filtered_german.append(re.sub(r'[^\w\s]', '', ger_sent))

    return filtered_english, filtered_german


def create_masks(eng_batch, de_batch, max_sequence_length):
    NEG_INFTY = -1e9
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
        try:
            # Sometimes there's no padding
            eng_end_idx = torch.where(eng_batch[idx] == 0)[0][0].item()
        except:
            eng_end_idx = max_sequence_length
        try:
            de_end_idx = torch.where(de_batch[idx] == 0)[0][0].item()
        except:
            de_end_idx = max_sequence_length
            
        eng_chars_to_padding_mask = np.arange(eng_end_idx+1, max_sequence_length)
        de_chars_to_padding_mask = np.arange(de_end_idx+1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, de_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, de_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, de_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    #print(f"encoder_self_attention_mask {encoder_self_attention_mask.size()}:\n {encoder_self_attention_mask[0, :10, :10]}")
    #print(f"decoder_self_attention_mask {decoder_self_attention_mask.size()}:\n {decoder_self_attention_mask[0, :10, :10]}")
    #print(f"decoder_cross_attention_mask {decoder_cross_attention_mask.size()}:\n {decoder_cross_attention_mask[0, :10, :10]}")
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]