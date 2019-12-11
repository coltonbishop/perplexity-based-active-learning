import lm
import sys

corpus = 'data/brown-train.txt'
tokenizer, model = lm.model(corpus)

n = 10
top_ten = lm.rank_sentences(model, tokenizer, corpus, 10)

print(top_ten)