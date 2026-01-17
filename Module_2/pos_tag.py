# avec nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

text = "Every part of our speech can be decomposed by tokens"
words = word_tokenize(text)

pos_tags = pos_tag(words)

print("\nresultat PoS tagging avec nltk:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")

#avec spacy

import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

print("resultat PoS tagging avec spacy:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
