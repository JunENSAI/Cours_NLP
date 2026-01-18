import spacy
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def custom_tokenizer(doc):
    return [t.text for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha]

train_corpus = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test_corpus = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


print('Train size:', len(train_corpus.data))
print('Test size:', len(test_corpus.data))

print('\nExample article:', train_corpus.data[0])

nlp = spacy.load('en_core_web_sm')

vectorizer = CountVectorizer(tokenizer=custom_tokenizer)

X = vectorizer.fit_transform(train_corpus.data)
y = train_corpus.target

X_test = vectorizer.transform(test_corpus.data)
y_test = test_corpus.target

clf = MultinomialNB()

clf.fit(X, y)

test_preds = clf.predict(X_test)

article = ["Nasa's Swot satellite will survey millions of rivers and lakes"]

X_article = vectorizer.transform(article)

proba_article = clf.predict_proba(X_article)

id_max = np.argmax(proba_article)
pred_article = train_corpus.target_names[id_max]
conf = np.max(proba_article)

print('Prediction:', pred_article)
print('Confidence:', conf)
