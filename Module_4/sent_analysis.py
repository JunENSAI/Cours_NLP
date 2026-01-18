import random
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

documents = []

for fileid in movie_reviews.fileids('pos'):
    documents.append((movie_reviews.raw(fileid), 1))
    
for fileid in movie_reviews.fileids('neg'):
    documents.append((movie_reviews.raw(fileid), 0))

random.seed(1001) 
random.shuffle(documents)

X = [d[0] for d in documents]
y = [d[1] for d in documents]

print(f"Nombre total de critiques : {len(X)}")
print(f"Exemple de critique :\n{X[0][:100]}...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

accuracy = accuracy_score(y_test, clf.predict(X_test_vec))
print(f"\nPrécision du modèle (Accuracy) : {accuracy:.2%}")

y_pred = clf.predict(X_test_vec)

print("--- Rapport de Classification ---")
print(classification_report(y_test, y_pred, target_names=['Négatif', 'Positif']))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Négatif', 'Positif'])
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion")
plt.show()

# essai de prédiction

my_review = ["The movie lacked special effects and the acting was not up to the standard of the budget allocated"]
pred = clf.predict(vectorizer.transform(my_review))
print(f"\nTest manuel : '{my_review[0]}'")
print(f"Résultat : {'POSITIF' if pred[0] == 1 else 'NÉGATIF'}")
