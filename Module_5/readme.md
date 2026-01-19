# Deep Learning Séquentiel (RNN & LSTM)

## 1. Padding, Truncation & Masking

Un réseau de neurones attend une entrée de taille fixe (ex: une matrice 32x100). Mais le langage naturel est de longueur variable. Le Padding est la technique qui permet de faire entrer des données variables dans une structure fixe.

### Concepts Clés

1. Le Token Spécial `<PAD>`

    C'est un "faux mot" ajouté au vocabulaire, généralement à l'index **0**.

    * Il ne signifie rien.

    * Il sert juste de "rembourrage" pour combler les trous.

2. Padding (Pre vs Post)

    Si on décide que toutes les phrases doivent faire 10 mots ($L=10$) :

    * **Post-Padding (Standard)** : "J'aime le NLP `[PAD]` `[PAD]` `[PAD]`". (On ajoute les zéros à la fin).

    * **Pre-Padding** : "`[PAD]` `[PAD]` `[PAD]` J'aime le NLP". (On ajoute au début).

    * *Note :* Pour les RNN, le Post-Padding est souvent préféré car le réseau traite le vrai contenu en premier, mais les RNN bidirectionnels (Bi-LSTM) gèrent bien les deux.

3. Truncation (Troncature)

    Si la phrase est plus longue que la limite fixée (ex: 15 mots alors que $L=10$) :

    * On coupe brutalement.

    * On perd de l'information, mais c'est nécessaire pour gérer la mémoire GPU.

    * Généralement, on tronque la fin (Post-Truncation), mais parfois le début si la fin de la phrase contient la conclusion importante.

4. Attention Mask (Le Masque)

    C'est un vecteur binaire envoyé au réseau en parallèle des données.

    * **1** : Ceci est un vrai mot -> Traite-le.

    * **0** : Ceci est du Padding -> Ignore-le (ne calcule pas de gradient dessus, ne le laisse pas influencer la moyenne).

    * Sans masque, le réseau risque d'apprendre que le mot "0" est très fréquent et influence le sens, ce qui est faux.

5. Dynamic Padding

    Au lieu de fixer une taille de 100 pour tout le dataset (alors que la plupart des phrases font 10 mots), on le fait **par batch**.

    * Si dans le Batch 1, la phrase la plus longue fait 12 mots -> Tout le batch est paddé à 12.

    * Si dans le Batch 2, la phrase la plus longue fait 50 mots -> Tout le batch est paddé à 50.

    * *Avantage :* Gain de vitesse énorme (on ne calcule pas des milliers de zéros inutiles).

---

## 2. La Couche d'Embedding

Dans le Module 3, nous avons vu Word2Vec, qui est un algorithme statique (on entraîne, on génère des vecteurs, et c'est fini).

En Deep Learning, l'Embedding est une **couche** du réseau de neurones.

### Concepts Clés

1. Lookup Table (Table de correspondance)

    Techniquement, une couche d'Embedding n'est pas une couche de calcul complexe (comme une multiplication matricielle dense). C'est un simple **dictionnaire géant**.

    * **Entrée :** Un index entier (ex: `452`, qui correspond au mot "chat").

    * **Opération :** Le réseau va chercher la ligne 452 dans sa matrice interne.

    * **Sortie :** Le vecteur stocké à cette ligne (ex: `[0.1, -0.5, ... ]`).

    * *C'est beaucoup plus rapide que de faire une multiplication "One-Hot encoding" x "Matrice de Poids".*

2. Poids Apprenables (Learnable Weights)

    C'est la différence majeure avec le Module 3.

    * Au début de l'entraînement, la couche d'Embedding est initialisée avec des **valeurs aléatoires** (bruit). Le mot "chat" n'a aucun sens.

    * Pendant la *Backpropagation*, le réseau modifie les valeurs de ces vecteurs pour minimiser l'erreur de prédiction finale.

    * **Conséquence :** Le réseau crée ses propres vecteurs, optimisés spécifiquement pour VOTRE tâche.

        * Si vous faites de l'analyse de sentiment, les mots "Excellent" et "Bon" vont se rapprocher mathématiquement.
        
        * Si vous faites de la classification grammaticale, "Excellent" se rapprochera de "Grand" (Adjectifs).

3. Transfer Learning (Pre-trained Embeddings)

    Vous n'êtes pas obligé de partir de zéro (Random). Vous pouvez **injecter** des vecteurs déjà entraînés (Glove, Word2Vec) dans cette couche.

    * **Fine-Tuning :** Vous chargez GloVe, et vous laissez le réseau modifier un peu les vecteurs pour s'adapter à votre jargon.

    * **Freezing :** Vous chargez GloVe et vous "gelez" la couche (rendez les poids non-modifiables). Le réseau doit se débrouiller avec ces vecteurs fixes.

4. Dimensions

    Une couche d'Embedding se définit par deux chiffres :

    * **Input Dim (Vocab Size) :** Combien de mots uniques existent (ex: 10 000). C'est le nombre de lignes de la table.

    * **Output Dim (Vector Size) :** La taille du vecteur (ex: 64, 128, 300). C'est le nombre de colonnes.

---

# 3. Réseaux de Neurones Récurrents (RNN)

Les algorithmes précédents (Naive Bayes, RegLog) traitaient le texte comme un "Sac de Mots". Ils ne savaient pas que le mot en position 1 venait avant le mot en position 2.

Les RNN (Recurrent Neural Networks) sont conçus pour traiter des **séquences** (Séries Temporelles, Texte, Audio).

### Concepts Clés

1. La Mémoire (Hidden State)

    Contrairement à un réseau classique qui est amnésique (il traite chaque entrée indépendamment), un RNN possède une **boucle de rétroaction**.
    * Au temps $t$, il prend deux entrées :
        1. Le mot actuel ($x_t$).

        2. Sa propre mémoire de l'étape précédente ($h_{t-1}$).

    * Il mélange ces deux informations pour produire une nouvelle mémoire ($h_t$).

    * *Analogie :* C'est comme lire une phrase. Pour comprendre le mot "il", votre cerveau utilise le contexte des mots lus juste avant.

2. Backpropagation Through Time (BPTT)

    Pour entraîner ce réseau, on le "déplie" (Unrolling) dans le temps.
    * Si votre phrase fait 10 mots, le réseau est copié 10 fois virtuellement.

    * L'erreur calculée à la fin de la phrase est propagée en arrière jusqu'au premier mot pour ajuster les poids.

3. Le Problème du "Vanishing Gradient"

    C'est la limite majeure qui a tué le RNN classique (SimpleRNN).

    * Lors de la rétropropagation sur une longue phrase (ex: 50 mots), le signal d'erreur (gradient) devient de plus en plus petit à chaque étape.

    * Arrivé au début de la phrase, le signal est si petit (proche de 0) que le réseau **n'apprend plus rien**.

    * *Conséquence :* Le RNN a une mémoire de poisson rouge. Il se souvient des derniers mots, mais oublie le début du paragraphe.

    * *Solution :* Les LSTM (Partie 2).

4. Input Shape (3D Tensor)

    Les RNN attendent des données en 3 dimensions, ce qui perturbe les débutants :

    1. **Batch Size** : Combien de phrases on traite en même temps (ex: 32).

    2. **Time Steps** : La longueur de la phrase (ex: 100 mots).

    3. **Features** : La taille du vecteur de chaque mot (ex: dimension de l'embedding).

---

## 4. LSTM (Long Short-Term Memory) & GRU

Le LSTM (inventé par Schmidhuber en 1997) et le GRU sont des architectures conçues pour avoir une "mémoire à long terme".

### Concepts Clés

1. La "Cell State"

    La grande différence du LSTM, c'est qu'il possède deux mémoires qui circulent :

    * **Hidden State ($h_t$)** : La mémoire à court terme (comme le RNN classique), utilisée pour la prédiction immédiate.

    * **Cell State ($c_t$)** : La mémoire à long terme. C'est une "autoroute" qui traverse tout le réseau avec très peu d'interactions. L'information peut y voyager intacte du début à la fin de la phrase.

2. Les Portes (Gates)

    Le LSTM utilise des mécanismes de régulation (des portes sigmoïdes entre 0 et 1) pour décider de modifier ou non la Cell State.

    1.  **Forget Gate** : "Est-ce que je garde l'info précédente ?" (Ex: Le sujet a changé, j'oublie le genre de l'ancien sujet).

    2.  **Input Gate** : "Est-ce que l'info actuelle est importante ?" (Ex: Je viens de lire un nouveau sujet important, je le stocke).

    3.  **Output Gate** : "Qu'est-ce que je révèle au reste du réseau ?"

3. GRU (Gated Recurrent Unit)

    C'est une version simplifiée du LSTM (2014).

    * Il fusionne la *Cell State* et le *Hidden State*.

    * Il a seulement 2 portes (Reset & Update).

    * **Avantage :** Plus rapide à entraîner et consomme moins de mémoire RAM.

    * **Performance :** Souvent équivalente au LSTM sur des tâches standards.

4. Bidirectionnalité (Bi-LSTM)

    En NLP, le contexte vient aussi de la *fin* de la phrase.

    * *Exemple :* Dans "L'avocat est...", le mot "avocat" est ambigu. Si la fin est "...délicieux", c'est le fruit. Si c'est "...au tribunal", c'est le métier.

    * Un **Bi-LSTM** lit la phrase dans les deux sens (Gauche->Droite et Droite->Gauche) et concatène les résultats. C'est la base des modèles modernes.

---
