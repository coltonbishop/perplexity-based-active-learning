import re
import string
import unicodedata
import os
import pickle
import nltk
from nltk import tokenize


# Word Frequency Counter (word->frequency)
frequency = {}
known = []
phrases = []
model = ""
# Loads stored data
pickle_in = open("freq.pickle","rb")
frequency = pickle.load(pickle_in)
pickle_in = open("known.pickle","rb")
known = pickle.load(pickle_in)
pickle_in = open("phrases.pickle","rb")
phrases = pickle.load(pickle_in)

# Reads in sources and updates frequency and known
def read_in(english_sources, translated_sources):
	global frequency, known, phrases

	# Reads in new source texts and stores
	for source in english_sources:
		add_to_freq(word_list(source))
		file_content = open(source).read().decode('utf8')
		tokens = tokenize.sent_tokenize(file_content)
		phrases.append(tokens)

	pickle_out = open("freq.pickle","wb")
	pickle.dump(frequency, pickle_out)
	pickle_out.close()
	pickle_out = open("phrases.pickle","wb")
	pickle.dump(phrases, pickle_out)
	pickle_out.close()
	# Reads in new Known Words and stores
	for source in translated_sources:
		add_to_known(word_list(source))
	pickle_out = open("known.pickle","wb")
	pickle.dump(known, pickle_out)
	pickle_out.close()

# Updates global frequency dictionary and known list given a list of words
def add_to_freq(word_list):
	global frequency, model
	#count = frequency.get(word,0)
	if model == "unigram":
		for word in word_list:
			if word not in frequency:
				frequency[word] = 1
			else:
				frequency[word] += 1
	if model == "bigram":
		last = word_list[0]
		for i in range(1, len(word_list)):
			current = word_list[i]
			key = (last, current)
			if key not in frequency:
				frequency[key] = 1
			else:
				frequency[key] += 1
			last = current
	if model == "trigram":
		last_last = word_list[0]
		last = word_list[1]
		for i in range(2, len(word_list)):
			current = word_list[i]
			key = (last_last, last, current)
			if key not in frequency:
				frequency[key] = 1
			else:
				frequency[key] += 1
			last_last = last
			last = current

# Adds new words from word list to known
def add_to_known(word_list, save=False):

	global known, model
	#count = frequency.get(word,0)
	if model == "unigram":
		for word in word_list:
			if word not in known:
				known.append(word)
	if model == "bigram":
		last = word_list[0]
		for i in range(1, len(word_list)):
			current = word_list[i]
			key = (last, current)
			if key not in known:
				known.append(key)
			last = current
	if model == "trigram":
		last_last = word_list[0]
		last = word_list[1]
		for i in range(2, len(word_list)):
			current = word_list[i]
			key = (last_last, last, current)
			if key not in known:
				known.append(key)
			last_last = last
			last = current
	if save:
		pickle_out = open("known.pickle","wb")
		pickle.dump(known, pickle_out)
		pickle_out.close()

# Clears all data structures
def clear():
	clear_freq()
	clear_phrases()
	clear_known()

# Clears the current dictionary of word frequencies
def clear_freq():
	global frequency
	frequency = {}
	pickle_out = open("freq.pickle","wb")
	pickle.dump(frequency, pickle_out)
	pickle_out.close()

# Clears the current dictionary of word frequencies
def clear_phrases():
	global phrases
	phrases = []
	pickle_out = open("phrases.pickle","wb")
	pickle.dump(phrases, pickle_out)
	pickle_out.close()

# Clears the current list of known words
def clear_known():
	global known
	known = []
	pickle_out = open("known.pickle","wb")
	pickle.dump(known, pickle_out)
	pickle_out.close()

def get_known():
	global known
	return known

def get_frequency():
	global frequency
	return frequency

def get_phrases():
	global phrases
	return phrases

# Returns a list of words given a text filename
def word_list(filename):
	document_text = open(filename, 'r')
	text_string = document_text.read().lower()
	unicode_string = unicode(text_string, "utf-8")
	nfkd_form = unicodedata.normalize('NFKD', unicode_string)
	only_ascii = nfkd_form.encode('ASCII', 'ignore')
	match_pattern = re.findall(r'\b[a-z\-]{0,15}\b', only_ascii)
	return match_pattern

# Prints the top x list of all words and their frequencies
def print_freq(x):
	global frequency
	for key, value in sorted(frequency.items(), key=lambda item: item[1], reverse=True):
		print("%s: %s" % (key, value))
		x -= 1
		if x == 0:
			break

# Returns the top X most critical words
# Critical words are frequent words that 
def critical_words(x):
	global frequency, known
	critical_words = []
	count = 0
	for w in sorted(frequency, key=frequency.get, reverse=True):
		if count == x:
			return critical_words
		if w not in known:
			critical_words.append(w)
			count = count + 1
	return critical_words

def convert_to_bigrams(phrase):
	bigrams = []
	last = phrase[0]
	for i in range(1, len(phrase)):
		current = phrase[i]
		bigrams.append((last,current))
		last = current
	return bigrams


def convert_to_trigrams(phrase):
	trigrams = []
	last_last = phrase[0]
	last = phrase[1]
	for i in range(2, len(phrase)):
		current = phrase[i]
		trigrams.append((last_last, last,current))
		last_last = last
		last = current
	return trigrams

# Returns a critical phrase to be translated
def critical_phrase(x=0):
	global phrases, model
	tally = 0
	count = x
	while(count < x + 10):
		word = critical_words(x+11)[count]
		for i in range(len(phrases)):
			for j in range(len(phrases[i])):
				if model == "unigram":
					converted_phrase = phrases[i][j]
				if model == "bigram":
					converted_phrase = convert_to_bigrams(phrases[i][j])
				if model == "trigram":
					converted_phrase = convert_to_trigrams(phrases[i][j])
				if word in converted_phrase:
					print(word)
					print(converted_phrase)
					if tally == x:
						# Later only add to known if user responds with translation
						add_to_known([word], save=True)
						return phrases[i][j]
					tally = tally + 1
		count = count + 1

def load_new_data(english_sources, translated_sources, model_type="unigram"):
	global model
	clear()
	# Large corpus of english Text for Which Transled Data Does Not Yet Exist
	# English Text for Which Transled Data Exists
	model = model_type
	read_in(english_sources, translated_sources)


