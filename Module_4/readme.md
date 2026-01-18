# Classification de Texte et Analyse de Sentiment

## 1. Naive Bayes

Naive Bayes est un algorithme probabiliste basé sur le théorème de Bayes. Il est utilisé en NLP pour sa rapidité et son efficacité sur des données textuelles (haute dimension), même avec peu d'exemples.

### Concepts Clés

1. Pourquoi "Naïf" ?

    L'algorithme est dit "naïf" parce qu'il fait une hypothèse très forte (et souvent fausse en linguistique) : **l'indépendance des traits**.
    * Il considère que la présence d'un mot n'a aucun lien avec la présence d'un autre.

    * *Exemple :* Pour lui, dans "New York", le mot "New" et le mot "York" sont traités séparément, sans lien de causalité.

    * *Malgré cette "fausseté", ça marche très bien en pratique.*

2. Le Théorème de Bayes (La Formule)

    Il permet d'inverser les probabilités. Nous voulons connaître la probabilité d'une **Classe (Y)** sachant le **Texte (X)**.

    $$P(Classe | Mot) = \frac{P(Mot | Classe) \times P(Classe)}{P(Mot)}$$

    * **Posterior ($P(C|X)$)** : Probabilité que le message soit un SPAM sachant qu'il contient "Viagra". (C'est ce qu'on cherche).

    * **Likelihood ($P(X|C)$)** : Probabilité de trouver le mot "Viagra" dans les messages qui sont DÉJÀ connus comme SPAM. (On apprend ça durant l'entraînement).

    * **Prior ($P(C)$)** : Probabilité qu'un message soit un SPAM en général (ex: 20% des mails sont des spams).

3. Le problème du Zéro (Laplace Smoothing)

    Imaginez que le mot "Casino" n'apparaisse jamais dans vos emails de "Travail".

    * Probabilité $P(\text{"Casino"} | \text{Travail}) = 0$.

    * Comme on multiplie les probabilités des mots, **un seul zéro annule tout le calcul**.

    * **Solution (Lissage de Laplace)** : On ajoute artificiellement +1 à tous les comptes de mots. Ainsi, aucun mot n'a jamais une probabilité nulle.

4. Les Variantes

    * **MultinomialNB** : Le standard pour le NLP. Il utilise la fréquence des mots (BoW ou TF-IDF). Il se soucie de *combien de fois* un mot apparaît.

    * **BernoulliNB** : Il ne regarde que la présence binaire (0 ou 1). Le mot est là ou pas ? Utile pour des textes très courts.

---

## 2. Régression Logistique

Ne vous fiez pas à son nom : la Régression Logistique est bien un algorithme de **Classification** (et non de régression). C'est le modèle linéaire de référence en industrie pour le NLP avant l'arrivée du Deep Learning.

### Concepts Clés

1. Discriminatif vs Génératif

    * **Naive Bayes (Génératif)** : Il essaie de modéliser à quoi *ressemble* un document "Sport" et à quoi *ressemble* un document "Cinéma".

    * **Régression Logistique (Discriminatif)** : Il ne s'intéresse pas à la structure du texte. Il cherche uniquement la **frontière** (une ligne ou un hyperplan) qui sépare le mieux les deux classes.

2. La Fonction Sigmoïde

    Le coeur de l'algorithme.

    * Un modèle linéaire classique produit un score qui peut aller de $-\infty$ à $+\infty$.

    * Pour faire de la classification, nous devons écraser ce score entre **0 et 1** (pour avoir une probabilité).

    * C'est le rôle de la **Sigmoïde** (courbe en S).

    $$P(y=1|x) = \frac{1}{1 + e^{-(wx + b)}}$$

3. Les Poids (Weights) = Explicabilité

    C'est le plus grand avantage de ce modèle. Il attribue un coefficient (poids) à chaque mot du vocabulaire.

    * **Poids Positif élevé** : Le mot vote fortement pour la Classe 1 (ex: "Football" pour la classe Sport).

    * **Poids Négatif élevé** : Le mot vote fortement pour la Classe 0 (ex: "Acteur" pour la classe Sport, car il indique plutôt le Cinéma).

    * **Poids proche de 0** : Le mot est inutile (ex: "le", "de", "aujourd'hui").

4. Régularisation (L1/L2)

    Les textes contiennent des milliers de mots (haute dimension). Le modèle risque d'apprendre par cœur le bruit (Overfitting).

    * La régression logistique applique une pénalité (Régularisation) pour forcer les poids des mots inutiles à rester petits, ce qui rend le modèle plus robuste sur des nouveaux textes.

---


