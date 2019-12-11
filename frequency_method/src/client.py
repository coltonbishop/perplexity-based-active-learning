import gap
import sys


# Scans through data to initialize data structures
english_sources = ["data/book1.txt", "data/book2.txt", "data/book3.txt"]
translated_sources = ["data/book4.txt", "data/book5.txt"]
gap.load_new_data(english_sources, translated_sources, model_type="unigram")

# Prints the x most frequent words (and their frequencies )
x = 3
gap.print_freq(x)

# Print critical phrases
# Once a critical phrase is proivided (and translated),
# the data structures are updated to represent this new knowledge
for k in range(0,x):
	print gap.critical_phrase()
