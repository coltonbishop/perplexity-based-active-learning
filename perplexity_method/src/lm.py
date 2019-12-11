""""""

import sys
import os
import numpy as np
import argparse
from tqdm import tqdm
from nltk import tokenize
from data import Tokenizer, readCorpus


class LanguageModel:
  def __init__(self, vocab, n=2, smoothing=None, smoothing_param=None):
    assert n >=2, "This code does not allow you to train unigram models."
    self.vocab = vocab
    self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
    self.n = n
    self.smoothing = smoothing    
    self.smoothing_param = smoothing_param
    self.bi_counts = None      # Holds the bigram counts
    self.bi_prob = None        # Holds the computed bigram probabilities.

    assert smoothing is None or smoothing_param is not None, "Forgot to specify a smoothing parameter?"


  """Compute basic bigram probabilities (without any smoothing)"""
  def computeBigramProb(self):
    self.bi_prob = self.bi_counts.copy()
    for i, _ in enumerate(tqdm(self.bi_prob, desc="Estimating bigram probabilities")):
      cnt = np.sum(self.bi_prob[i])
      if cnt > 0:
        self.bi_prob[i] /= cnt
        
  ### TODO: complete ###
  def computeBigramProbAddAlpha(self, alpha=0.001):   
    self.bi_prob = self.bi_counts.copy()
    self.bi_prob = np.add(self.bi_prob, np.full((len(self.vocab), len(self.vocab)), alpha))
    for i, _ in enumerate(tqdm(self.bi_prob, desc="Estimating bigram probabilities")):
      cnt = np.sum(self.bi_prob[i])
      if cnt > 0:
        self.bi_prob[i] /= cnt
    print(self.bi_prob[i])

  ### TODO: complete ###
  def computeBigramProbInterpolation(self, beta=0.01):
    print("PARAMETER IS {}".format(self.smoothing_param))
    alpha = 0.001
    self.bi_prob = self.bi_counts.copy()
    self.bi_prob = np.add(self.bi_prob, np.full((len(self.vocab), len(self.vocab)), alpha))
    # uni = []
    total = np.sum(self.bi_prob)
    # for i in range(0, len(self.bi_prob[0])):
    #   uni.append(np.sum(self.bi_prob[:,i])/total)
    for i, _ in enumerate(tqdm(self.bi_prob, desc="Estimating bigram probabilities")):
      cnt = np.sum(self.bi_prob[i])
      if cnt > 0:
        self.bi_prob[i] /= cnt
        self.bi_prob[i] *= beta
        self.bi_prob[i] += (1-beta)*(cnt/total) # Ask about this in office hours
    return

  """Train a basic n-gram language model"""
  def train(self, corpus):
    if self.n==2:
      self.bi_counts = np.zeros((len(self.vocab), len(self.vocab)), dtype=float)
    else:
      raise ValueError("Only bigram model has been implemented so far.")
    
    # Convert to token indices.
    corpus = [self.token_to_idx[w] for w in corpus]

    # Gather counts
    for i, idx in enumerate(tqdm(corpus[:1-self.n], desc="Counting")):
      self.bi_counts[idx][corpus[i+1]] += 1

    # Pre-compute the probabilities.
    if not self.smoothing:
      self.computeBigramProb()
    elif self.smoothing == 'addAlpha':
      self.computeBigramProbAddAlpha(self.smoothing_param)
    elif self.smoothing == 'interpolation':
      self.computeBigramProbInterpolation(self.smoothing_param)
    else:
      raise ValueError("Unknown smoothing type.")


  def test(self, corpus):
    logprob = 0.

    # Convert to token indices.
    corpus = [self.token_to_idx[w] for w in corpus]

    for i, idx in enumerate(tqdm(corpus[:1-self.n], desc="Evaluating")):
      logprob += np.log(self.bi_prob[idx, corpus[i+1]])

    # import pdb; pdb.set_trace()

    logprob /= len(corpus[:1-self.n])

    # Compute perplexity
    ppl = np.exp(-logprob)

    return ppl

  def test_ppl(self, corpus):
    logprob = 0.

    # Convert to token indices.
    corpus = [self.token_to_idx[w] for w in corpus]

    for i, idx in enumerate(tqdm(corpus[:len(corpus)-1], desc="Evaluating")):
      logprob += np.log(self.bi_prob[idx, corpus[i+1]])

    # import pdb; pdb.set_trace()
    
    logprob /= len(corpus)

    # Compute perplexity
    ppl = np.exp(-logprob)

    return ppl

def model(corpus):
  tokenizer = Tokenizer(tokenize_type='basic', lowercase=True)
  train_corpus = readCorpus(corpus, tokenizer)
  lm = LanguageModel(tokenizer.vocab, n=2, smoothing='addAlpha', smoothing_param=0.001)
  train_idx = int(1.0 * len(train_corpus))
  lm.train(train_corpus[:train_idx])

  return tokenizer, lm

# Ranks sentences based on perplexity
def rank_sentences(lm, tokenizer, corpus, n):
  with open(corpus) as f:
    sentences = tokenize.sent_tokenize(f.read())
  # train_ppl = lm.test(train_corpus[:train_idx])
  # val_ppl = lm.test(val_corpus)

  # print("Train perplexity: %f, Val Perplexity: %f" %(train_ppl, val_ppl))

  phrases = []

  s = 0
  for sentence in sentences:
    sent_tokens = tokenizer.tokenize(sentence)
    print(sent_tokens)
    ppl = lm.test_ppl(sent_tokens)
    phrases.append((ppl, sentence))

  phrases.sort()
  phrases.reverse()

  return phrases[0:n]


def main(args):
  print(args)
  #tokenizer = Tokenizer(tokenize_type='nltk', lowercase=True)
  #tokenizer = Tokenizer(tokenize_type='basic', lowercase=True)
  tokenizer = Tokenizer(tokenize_type='basic', lowercase=True)

  train_corpus = readCorpus(args.train_file, tokenizer)
  #val_corpus = readCorpus(args.val_file, tokenizer)

  # Print the top 10 words in the corpus. 
  # IMPORTANT: complete the function within the tokenizer class in data.py first.
  #print("Top 10 words: %s" %(tokenizer.countTopWords(train_corpus, k=10)))

  #tokenizer.plot(train_corpus)

  # Instantiate the language model.
  lm = LanguageModel(tokenizer.vocab, n=2, smoothing=args.smoothing, smoothing_param=args.smoothing_param)

  # Figure out index for a specific percentage of train corpus to use. 
  train_idx = int(args.train_fraction * len(train_corpus))

  print("TRAINING FRACTION IS {}".format(args.train_fraction))

  lm.train(train_corpus[:train_idx])





if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Input data
  parser.add_argument('--train_file', default='data/brown-train.txt')
  parser.add_argument('--val_file', default='data/brown-val.txt')
  parser.add_argument('--train_fraction', type=float, default=1.0, help="Specify a fraction of training data to use to train the language model.")
  parser.add_argument('--smoothing', type=str, default=None, help="Specify smoothing to use, if any.", choices=[None, 'addAlpha', 'interpolation'])
  parser.add_argument('--smoothing_param', type=float, default=0.01, help="Specify smoothing parameter if using smoothing (i.e. value of alpha for addAlpha, beta for interpolation).")
  
  args = parser.parse_args()
  main(args)