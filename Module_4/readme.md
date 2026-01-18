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

## 3. Analyse de Sentiment (Opinion Mining)

L'analyse de sentiment consiste à déterminer l'attitude émotionnelle d'un auteur vis-à-vis d'un sujet. Le but est généralement de classer le texte selon une **Polarité** : Positif, Négatif ou Neutre.

### Concepts Clés

1. Approche Lexicale (Rule-based / Lexicon)

    C'est la méthode "sans apprentissage". On utilise un dictionnaire pré-défini de mots avec des scores.

    * *Exemple :* "Excellent" (+2.0), "Bon" (+1.0), "Horrible" (-2.0).

    * On additionne les scores des mots de la phrase.

    * **Avantage :** Pas besoin de données d'entraînement (Dataset). Rapide.

    * **Inconvénient :** Ne comprend pas le contexte complexe.

2. Approche Machine Learning (Supervised)

    C'est la méthode vue dans les parties précédentes (Naive Bayes, RegLog).

    * On donne au modèle 10 000 critiques de films étiquetées (Positif/Négatif).

    * Il apprend quels mots (et combinaisons de mots) prédisent le sentiment.

    * **Avantage :** Plus précis sur des domaines spécifiques.

3. Défis Majeurs

    L'analyse de sentiment échoue souvent à cause de nuances linguistiques :

    * **La Négation :** "Ce film n'est **pas** mauvais." (Les mots sont négatifs, mais le sens est positif).

    * **L'Ironie / Sarcasme :** "Super, mon téléphone est encore cassé !" (Le mot "Super" est positif, mais le sens est négatif).

    * **L'Ambiguïté :** "Le film est imprévisible." (Positif pour un thriller, Négatif pour une comédie romantique).

4. Subjectivité vs Polarité

    Certains outils distinguent deux métriques :

    * **Polarité :** De -1 (Négatif) à +1 (Positif).

    * **Subjectivité :** De 0 (Fait objectif) à 1 (Opinion personnelle).

    * *Exemple :* "La table est rouge" (Neutre, Objectif). "J'aime cette table rouge" (Positif, Subjectif).

### Datasets Standards

Pour entraîner un modèle d'analyse de sentiment, on utilise des corpus annotés (où un humain a déjà dit "Ceci est positif").

* **IMDb Movie Reviews** : Le standard académique. 25 000 critiques positives, 25 000 négatives. Vocabulaire très riche et argotique.

* **Amazon Product Reviews** : Plus orienté "objet" et "utilité".

* **Twitter Sentiment140** : Très court, beaucoup d'abréviations, d'emojis et de fautes.

### Le défi du "Domain Shift"

* Un modèle entraîné sur des **critiques de films** (ex: "Le scénario est lent" = Négatif) fonctionnera très mal sur des **critiques de restaurants** (ex: "Le service est lent" = Négatif, mais "On a mangé lentement" = Positif ?).

* Le vocabulaire du sentiment change selon le contexte. On ne peut pas aveuglément appliquer un modèle IMDb sur des données bancaires.

---

## 4. Métriques d'Évaluation

Une fois le modèle entraîné, il faut mesurer sa performance. La métrique "Accuracy" (Précision globale) est souvent trompeuse, surtout si les données sont déséquilibrées (ex: 90% de classes Positives et 10% de Négatives).

## Concepts Clés

1. `La Matrice de Confusion`

    C'est le tableau de bord fondamental qui compare les **Prédictions** du modèle avec la **Vérité Terrain** (Réalité).

    Il y a 4 cas possibles :

    * **Vrais Positifs (TP)** : Le modèle a prédit "Positif" et c'était Vrai.

    * **Vrais Négatifs (TN)** : Le modèle a prédit "Négatif" et c'était Vrai.

    * **Faux Positifs (FP)** : Le modèle a prédit "Positif" mais c'était Faux (Erreur de Type I - "Fausse Alerte").

    * **Faux Négatifs (FN)** : Le modèle a prédit "Négatif" mais c'était Positif (Erreur de Type II - "Raté").

2. `Précision` (Precision) - "La Qualité"

    *Question :* "Quand mon modèle affirme que c'est Positif, a-t-il raison ?"

    * *Formule :* $TP / (TP + FP)$

    * *Utilité :* Crucial pour un filtre anti-spam (on ne veut surtout pas mettre un mail important dans les spams -> on veut 0 Faux Positifs).

3. `Rappel` (Recall) - "La Quantité"

    *Question :* "Sur tous les cas qui étaient réellement Positifs, combien mon modèle en a-t-il trouvé ?"

    * *Formule :* $TP / (TP + FN)$

    * *Utilité :* Crucial pour la médecine (détection de cancer). On préfère avoir une fausse alerte (FP) plutôt que de rater un malade (FN).

4. `F1-Score`

    C'est la moyenne harmonique entre la Précision et le Rappel.

    * Il punit les scores extrêmes. Si vous avez une très bonne Précision mais un Rappel nul, le F1-Score sera mauvais.

    * C'est la métrique reine pour comparer deux modèles.

---
