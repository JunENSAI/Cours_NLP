# implementation avec nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "Your key does not open the door"

stop_words = set(stopwords.words('english')) # ensemble de stopwords en anglais
tokens = word_tokenize(text.lower())

filtered_tokens = [word for word in tokens if word not in stop_words]

print("Original:", tokens)
print("Filtered with nltk:", filtered_tokens)

# implémentation avec Spacy

import spacy

# python3 -m spacy download en_core_web_sm : telecharge le modèle voulu
nlp = spacy.load("en_core_web_sm")
doc = nlp("Your key does not open the door")

filtered_words = [token.text for token in doc if not token.is_stop]
print("Filtered with spacy:", filtered_words)

# implementation avec gensim

from gensim.parsing.preprocessing import remove_stopwords


new_text = "Your key does not open the door"

new_filtered_text = remove_stopwords(new_text)

print("Original Text:", new_text)
print("Filtered with gensim:", new_filtered_text)