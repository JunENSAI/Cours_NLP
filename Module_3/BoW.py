# manuellement

corpus = [
    "Monster beneath your bed",
    "What is real",
    "Sweet heat lightning"
]

def bow(corpus):
    vocab = sorted(list(set(" ".join(corpus).lower().split())))
    
    vectors = []
    for sentence in corpus:
        tokens = sentence.lower().split()
        vec = [0] * len(vocab) 
        for word in tokens:
            if word in vocab:
                idx = vocab.index(word)
                vec[idx] += 1
        vectors.append(vec)
    
    return vocab, vectors

v, m = bow(corpus)
print(f"Vocab: {v}")
print(f"Matrix: {m}")

# avec scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

# Convertir en array numpy pour affichage (X est une matrice sparse par défaut)
print("Vocabulaire (Features) :", vectorizer.get_feature_names_out())
print("\nMatrice BoW :\n", X.toarray())

df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df)

