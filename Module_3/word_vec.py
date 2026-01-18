import json
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

corpus_text = data["corpus"]

tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus_text]

model = Word2Vec(sentences=tokenized_corpus, vector_size=20, window=5, min_count=1, workers=4)

print("\nMots les plus proches de 'learning':")
print(model.wv.most_similar('learning', topn=5))