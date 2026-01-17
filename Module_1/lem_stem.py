from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# perform stemming

stemmer = PorterStemmer()

words = ["learning", "comptutationally", "engineering"]

stemmed_words = [stemmer.stem(word) for word in words]

print(f"resultats après stemming : {stemmed_words}")

# perform lemmatization

lemmatizer = WordNetLemmatizer()
words = ["running", "better", "studies", "was"]
lemmas = [
    lemmatizer.lemmatize("running", pos=wordnet.VERB),
    lemmatizer.lemmatize("better", pos=wordnet.ADJ),
    lemmatizer.lemmatize("studies", pos=wordnet.NOUN),
    lemmatizer.lemmatize("was", pos=wordnet.VERB)
]
print(f"resultat lemmatization : {lemmas}")