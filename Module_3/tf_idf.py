import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(tfidf_matrix, vectorizer, doc_index=0):
    """
    Fonction qui sert à extraire les mots clés dans un doc contenu dans un corpus
    
    Args :
        tfidf_matrix : matrice de score tfidf calculé à partir du corpus
        vectorizer : objet de la classe TfidfVectorizer
        doc_index : position du doc dans le corpus
    
    return :
        df.sort_values(by=["tfidf"], ascending=False) : score tfidf ordonné de manière decroissante
        
    """
    feature_names = vectorizer.get_feature_names_out()
    
    doc_vector = tfidf_matrix[doc_index]    
    df = pd.DataFrame(doc_vector.T.todense(), index=feature_names, columns=["tfidf"])
    
    return df.sort_values(by=["tfidf"], ascending=False)

corpus = [
    "Everything is theoretically impossible until it is done",           
    "Every brilliant experiment like every great work of art starts with an act of imagination",
    "What you learn from a life in science is the vastness of our ignorance",
    "If I have seen further it is by standing on the shoulders of Giants",
    "Our virtues and our failures are inseparable like force and matter",           
    "Impossible only means that you have not found the solution yet"
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

df = pd.DataFrame(
    X.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=[f"Doc{i+1}" for i in range(len(corpus))]
)

print("Matrice TF-IDF :")
print(df)

# voir les mots clés dans le doc 2 du corpus selon le score tfidf
print("\nMots-clés du Document 2 :")
print(extract_keywords(X, tfidf, 1).head(3))
