from nltk.tokenize import word_tokenize
import os
from nltk.corpus import stopwords

class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text, stop_word = False):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab = set()
        self.stop = stop_word
        self.stoi = {}
        self.itos = {}
        self.stop_words = set(stopwords.words('english'))
        self.build_vocab(text)
       
        

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = word_tokenize(text)
        filtered_tokens = tokens
        if self.stop:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        self.vocab = set(filtered_tokens)
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = word_tokenize(text)
        filtered_tokens = tokens
        if self.stop:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
        return [self.stoi.get(word, self.stoi['<unk>']) for word in filtered_tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])
    
