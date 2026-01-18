# Représentation Vectorielle (Vectorisation & Embeddings)

## 1. Bag of Words (BoW)

Le modèle Bag of Words (Sac de mots) est la technique de vectorisation la plus élémentaire. Elle transforme un texte en un vecteur de longueur fixe en comptant la fréquence des mots.

### Concepts Clés

1. Le "Sac" (The Bag)

    Pourquoi appelle-t-on cela un "sac" ? Parce que **l'ordre des mots est perdu**. La structure grammaticale disparaît.

    * *Exemple :* "Le chien mord l'homme" et "L'homme mord le chien" produisent exactement le même vecteur BoW, car ils contiennent les mêmes mots avec les mêmes fréquences.

2. Le Vocabulaire

    C'est l'ensemble de tous les mots uniques présents dans tout le corpus (tous vos documents).

    * La taille du vecteur final est égale à la taille du vocabulaire ($N$).

    * Chaque position dans le vecteur correspond à un mot précis du vocabulaire.

3. Matrice Creuse (Sparse Matrix)

    * **Problème :** Si votre vocabulaire contient 20 000 mots, chaque phrase sera représentée par un vecteur de 20 000 dimensions. Cependant, une phrase n'utilise que 10 ou 20 mots.

    * **Résultat :** Le vecteur est rempli à 99% de zéros. C'est ce qu'on appelle une représentation "creuse" (sparse). Cela gaspille de la mémoire si on ne le gère pas correctement.

4. N-Grams (La solution partielle à la perte d'ordre)

    Pour capturer un peu de contexte (ex: la différence entre "pas bon" et "bon"), on peut utiliser des N-Grams.
    * **Unigram (1-gram)** : "Je", "suis", "content" (BoW classique).

    * **Bigram (2-gram)** : "Je suis", "suis content".

    * *Note :* Cela augmente considérablement la taille du vocabulaire.

5. Limitations principales

    * **Perte de sémantique :** "Voiture" et "Automobile" sont vus comme deux mots totalement différents (aucune relation mathématique entre eux).

    * **Fléau de la dimensionnalité :** De très grands vocabulaires rendent l'apprentissage difficile pour les modèles statistiques.

---

## 2. TF-IDF (Term Frequency - Inverse Document Frequency)

Le TF-IDF est une technique statistique qui convertit le texte en vecteurs, mais contrairement au Bag of Words, elle **pondère** l'importance des mots. Elle ne dit pas "combien de fois ce mot apparaît", mais "à quel point ce mot est **unique et important** pour ce document spécifique".

### Concepts Clés

1. L'intuition (Le problème du "LE" et du "DE")

    Dans le modèle Bag of Words, les mots les plus fréquents (comme "le", "est", "de") ont les plus grosses valeurs. Pourtant, ils portent très peu de sens.

    * **TF-IDF résout cela** : Il écrase le score des mots omniprésents et booste le score des mots rares.

2. TF (Term Frequency) - "Importance Locale"

    Mesure la fréquence d'un mot dans **un seul document**.

    * *Formule simplifiée :* 
    $$\text{TF}(t, d) = \frac{\text{Nombre d'occurrences du terme } t}{\text{Nombre total de mots dans le document } d}$$

    * Plus le mot apparaît dans le document, plus le score monte.

3. IDF (Inverse Document Frequency) - "Importance Globale"

    C'est le filtre de rareté. Il regarde l'ensemble du **corpus**.
    * *Formule :* $$\text{IDF}(t) = \log(\frac{N}{\text{df}(t)})$$
        * $N$ : Nombre total de documents.

        * $\text{df}(t)$ : Nombre de documents contenant le terme $t$.

    * **Mécanique clé :**
        * Si un mot est présent dans **tous** les documents (ex: "le"), $\log(1) = 0$. Le score s'annule.

        * Si un mot est très **rare** (ex: "hypoténuse"), le dénominateur est petit, le log est grand -> Le score explose.

4. Le Score Final
    $$w_{i,j} = \text{TF}_{i,j} \times \text{IDF}_i$$
    Un score élevé est atteint uniquement si le mot est **fréquent dans ce document précis** MAIS **rare ailleurs**.

5. Cas d'usage idéal

    Le TF-IDF est excellent pour :

    * **Moteurs de recherche** : Trouver le document le plus pertinent pour une requête spécifique.

    * **Extraction de mots-clés** : Identifier les mots qui résument un texte.

---

## 3. Word Embeddings & Word2Vec

Jusqu'à présent (BoW, TF-IDF), les mots étaient des cases isolées dans un tableau. "Roi" et "Reine" n'avaient aucun lien mathématique.
Les **Word Embeddings** (Plongements de mots) résolvent cela en transformant chaque mot en un **vecteur dense** de coordonnées dans un espace géométrique.

### Concepts Clés

1. Dense vs Sparse (Creux vs Dense)
    * **Avant (BoW)** : Vecteurs "Creux" (Sparse). Taille = 20,000 (vocabulaire). Rempli de zéros. Pas de sens.

    * **Maintenant (Embeddings)** : Vecteurs "Denses". Taille fixe et réduite (ex: 100 ou 300 dimensions). Rempli de nombres décimaux (ex: `[0.24, -0.45, 0.11...]`). Chaque chiffre représente une caractéristique abstraite du sens.

2. Sémantique Géométrique (Proximité = Sens)

    Dans cet espace vectoriel, la distance entre deux points représente leur similarité sémantique.
    * Si deux vecteurs sont proches géométriquement, les mots ont un sens similaire (ex: *Voiture* sera collé à *Camion*).

    * Le modèle apprend cela tout seul en regardant le **contexte** (les mots voisins).

3. Arithmétique des mots (Analogies)

    C'est une propriété sympa de Word2Vec. On peut faire des maths avec le sens :

    * $\text{Vecteur(Roi)} - \text{Vecteur(Homme)} + \text{Vecteur(Femme)} \approx \text{Vecteur(Reine)}$

    * $\text{Vecteur(Paris)} - \text{Vecteur(France)} + \text{Vecteur(Italie)} \approx \text{Vecteur(Rome)}$

4. Les Architectures (Comment ça apprend ?)

    Il existe deux méthodes principales pour entraîner Word2Vec (créé par Google en 2013) :
    * **CBOW (Continuous Bag of Words)** : Le modèle essaie de deviner le **mot caché au milieu** à partir des mots qui l'entourent.
        * *Entrée :* "Le", "chat", "___", "la", "souris". -> *Cible :* "mange".

    * **Skip-Gram** : L'inverse. À partir d'un mot central, il essaie de deviner les **mots alentours**.

        * *Entrée :* "mange". -> *Cible :* "chat", "souris".
        
        * *Note :* Skip-Gram est souvent meilleur pour les petits corpus et les mots rares.

5. OOV (Out of Vocabulary)

    La limite majeure de Word2Vec : si un mot n'était pas dans l'entraînement (ex: un néologisme ou une faute de frappe), le modèle ne peut pas le vectoriser.

### Implementation

```python
from gensim.models import Word2Vec

model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=10, min_count=1, workers=4)
```
Les paramètres de la fonction Word2Vec :

* `sentences` : resultante d'un tokenization par mots ou par phrases selon le contexte

* `vector_size` : définit la dimensionnalité des vecteurs de mots. Chaque mot sera représenté par un vecteur de 50 nombres.

* `windows` :  taille de la fenêtre de contexte. Le modèle considère 10 mots avant et 10 mots après chaque mot cible.

* `min_count` : seuil de fréquence minimum. Ignore les mots qui apparaissent moins de 1 fois.

* `workers` : nombre de threads CPU pour l'entraînement parallèle. Utilise 4 coeurs de processeur simultanément

---

# 4. Similarité Cosinus (Cosine Similarity)

Maintenant que nous avons transformé nos textes en vecteurs (via BoW, TF-IDF ou Embeddings), comment savoir si deux textes parlent de la même chose ?
On ne mesure pas la distance "physique" (Euclidienne) entre les points, mais **l'angle** entre les vecteurs.

### Concepts Clés

1. **Pourquoi pas la distance Euclidienne ?**

    La distance euclidienne est une mesure qui indique à quel point deux points sont éloignés dans un espace. C’est la longueur du segment droit qui relie deux points.

    * **Problème :** Si le Document A parle de "Science" (100 mots) et le Document B parle de "Science" (1000 mots), leurs vecteurs pointeront dans la même direction, mais le vecteur B sera beaucoup plus long (magnitude plus grande). La distance Euclidienne dira qu'ils sont très éloignés.

    * **Solution Cosinus :** La similarité cosinus regarde l'angle. Si les deux pointent dans la même direction, l'angle est nul, ils sont considérés identiques, peu importe la longueur du texte.

2. **Intuition**

    Imaginez deux flèches partant de l'origine (0,0).

    * **0° (Angle nul)** : Les flèches sont superposées. $\cos(0) = 1$. Similarité maximale (Textes identiques).

    * **90° (Angle droit)** : Les flèches sont perpendiculaires. $\cos(90) = 0$. Aucune corrélation (Les textes n'ont rien en commun).

    * **180° (Opposé)** : Les flèches sont opposées. $\cos(180) = -1$. Sens opposé (Rare en BoW/TF-IDF car les fréquences ne sont pas négatives, mais possible avec Word2Vec).

3. **Formule**

    C'est le produit scalaire des vecteurs divisé par le produit de leurs normes (longueurs).

    $$\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2} \times \sqrt{\sum B_i^2}}$$

4. **Applications**
    * **Moteur de recherche** : "Query" utilisateur vs "Documents" de la base. On trie par score Cosinus décroissant.

    * **Système de recommandation** : "Si tu as lu l'article A, voici l'article B qui a le score Cosinus le plus proche."

    * **Détection de plagiat** : Comparer le devoir d'un étudiant avec une base de données.

---
