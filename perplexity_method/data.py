
import os
import numpy as np
import nltk
from collections import Counter
import matplotlib.pyplot as plt

class Tokenizer:
  def __init__(self, tokenize_type='basic', lowercase=False):
    self.lowercase = lowercase  # If this is True, convert text to lowercase while tokenizing.
    self.type = tokenize_type
    self.vocab = []
    

  """This simple tokenizer splits the text using whitespace."""
  def basicTokenize(self, string):
    words = string.split()
    return words

  ### TODO : Fill in this function to use NLTK's word_tokenize() function. ###
  def nltkTokenize(self, string):
    return nltk.word_tokenize(string)

  def tokenize(self, string):
    if self.lowercase:
      string = string.lower()
    if self.type == 'basic':
      tokens = self.basicTokenize(string)
    elif self.type == 'nltk':
      tokens = self.nltkTokenize(string)
    else:
      raise ValueError('Unknown tokenization type.')    

    # Populate vocabulary
    self.vocab += list(set(tokens))

    return tokens

  def plot(self, words):
    count = Counter(words)
    c = count.values()
    c.sort()
    c.reverse()

    x = []
    for i in range(1, len(c)+1):
      x.append(i)

    plt.plot(x, c, 'ro')
    plt.axis([0, len(c)+2, 0, 61020])
    plt.show()


  ### TODO: Fill in this function to return the top k words (and their frequency) in the corpus according to frequency. ###
  def countTopWords(self, words, k):
    count = Counter(words)
    return(count.most_common(k))


def readCorpus(filename, tokenizer):  
  with open(filename) as f:
    words = tokenizer.tokenize(f.read())
  return words
