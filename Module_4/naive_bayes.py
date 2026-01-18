import spacy
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def custom_tokenizer(doc):
    return [t.text for t in nlp(doc) if not t.is_punct and not t.is_space and t.is_alpha]

train_corpus = fetch_20newsgroups(
    subset='train', 
    remove=('headers', 'footers', 'quotes'),
    categories=['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']
)
test_corpus = fetch_20newsgroups(
    subset='test', 
    remove=('headers', 'footers', 'quotes'),
    categories=['sci.space', 'rec.sport.baseball', 'comp.graphics', 'talk.politics.misc']
)

train_corpus.data = train_corpus.data[:1500]
train_corpus.target = train_corpus.target[:1500]

test_corpus.data = test_corpus.data[:400]
test_corpus.target = test_corpus.target[:400]

print('Train size:', len(train_corpus.data))
print('Test size:', len(test_corpus.data))

print('\nExample article:', train_corpus.data[0])

vectorizer = CountVectorizer(
    tokenizer=custom_tokenizer, 
    max_features=1500,
    token_pattern=None 
)

X = vectorizer.fit_transform(train_corpus.data)
y = train_corpus.target

X_test = vectorizer.transform(test_corpus.data)
y_test = test_corpus.target

clf = MultinomialNB()

clf.fit(X, y)

test_preds = clf.predict(X_test)

accuracy = np.mean(test_preds == y_test)
print(f'\nTest accuracy: {accuracy:.2%}')

article = ["Nasa's Swot satellite will survey millions of rivers and lakes"]

X_article = vectorizer.transform(article)

proba_article = clf.predict_proba(X_article)

id_max = np.argmax(proba_article)
pred_article = train_corpus.target_names[id_max]
conf = np.max(proba_article)

print('\nPrediction:', pred_article)
print('Confidence:', conf)