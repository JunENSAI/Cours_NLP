# Deep Learning Séquentiel (RNN & LSTM)

# 1. Réseaux de Neurones Récurrents (RNN)

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

# 2. LSTM (Long Short-Term Memory) & GRU

Le LSTM (inventé par Schmidhuber en 1997) et le GRU sont des architectures conçues pour avoir une "mémoire à long terme".

## Concepts Clés

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

