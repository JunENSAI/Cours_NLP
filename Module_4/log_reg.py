import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

categories = ['sci.space', 'rec.autos']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print(f"Documents chargés : {len(newsgroups_train.data)}")
print(f"Exemple de classe : 0={newsgroups_train.target_names[0]}, 1={newsgroups_train.target_names[1]}")

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

clf = LogisticRegression()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"\nPrécision (Accuracy) sur le test set : {score:.2%}")

feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]

top_positive = np.argsort(coefs)[-10:]
top_negative = np.argsort(coefs)[:10]

print("\nMots clés pour 'rec.autos' (Poids Négatifs) :")
print([feature_names[i] for i in top_negative])

print("\nMots clés pour 'sci.space' (Poids Positifs) :")
print([feature_names[i] for i in top_positive])

sample = ["Tesla vehicles run entirely on electricity and not on diesel fuel."]
pred = clf.predict(vectorizer.transform(sample))
print(f"\nPhrase : '{sample[0]}'")
print(f"Prédiction : {newsgroups_train.target_names[pred[0]]}")