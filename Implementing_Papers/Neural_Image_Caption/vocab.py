import nltk
import pickle
from collections import Counter

class Vocabulary(object):
    """Class to represent a vocabulary."""

    def __init__(self):
        self.word2idx = {}      # Dictionary mapping words to their indexes
        self.idx2word = {}      # Dictionary mapping indexes to their words
        self.idx = 0            # Starting index to begin assigning to words

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(dataframe, threshold=4):
    """Build vocab wrapper from pandas dataframe."""

    counter = Counter()

    for caption in dataframe['caption']:
        tokens = nltk.tokenize.word_tokenize(caption)
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab