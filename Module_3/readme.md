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

