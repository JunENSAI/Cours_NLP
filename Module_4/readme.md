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