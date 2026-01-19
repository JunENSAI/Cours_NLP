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

