import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = data["corpus"]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

df_sim = pd.DataFrame(cosine_sim, index=[f"Doc{i+1}" for i in range(len(documents))], columns=[f"Doc{i+1}" for i in range(len(documents))])
print("Matrice de Similarité :\n")
print(df_sim)