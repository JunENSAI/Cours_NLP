# Transformers & Mécanismes d'Attention

## 1. Le Mécanisme de Self-Attention

Les RNN lisaient mot par mot et oubliaient. L'Attention permet au modèle de regarder **toute la phrase en même temps** et de décider sur quels mots se "concentrer" pour comprendre le sens d'un mot spécifique.

### Concepts Clés

1. L'Analogie de la Base de Données (Query, Key, Value)

    Pour chaque mot de la phrase, le modèle génère 3 vecteurs différents :

    * **Query (Q - La Requête)** : Ce que le mot *cherche*. (Ex: Le mot "Il" cherche son antécédent).

    * **Key (K - La Clé)** : Ce que le mot *est*. (Ex: "Animal" s'identifie comme un nom masculin singulier).

    * **Value (V - La Valeur)** : Le *sens* intrinsèque du mot (ce qu'on veut extraire).

    *Le processus :* Le mot "Il" envoie sa Query. Elle matche fortement avec la Key de "Animal". Donc, on récupère beaucoup de la Value de "Animal".

2. Le Score d'Attention (Dot Product)

    Comment savoir si Q et K matchent ? On fait un **Produit Scalaire** (Dot Product).

    * Si les vecteurs sont alignés, le score est élevé.

    * Si les vecteurs sont orthogonaux, le score est nul.
    * Cela produit une matrice de scores qui dit : "À quel point le mot A est lié au mot B".

3. La Formule (Scaled Dot-Product Attention)

    C'est la formule enoncé dans le papier de 2017 :

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$$

    1.  $Q \times K^T$ : On calcule les similarités entre tous les mots.

    2.  $\div \sqrt{d_k}$ (**Scaling**) : On divise par la racine carrée de la dimension pour éviter que les chiffres soient trop grands (ce qui ferait exploser les gradients).

    3.  $\text{softmax}$ : On normalise pour que la somme des attentions soit égale à 1 (100%).

    4.  $\times V$ : On crée le nouveau vecteur du mot comme une somme pondérée des Valeurs des autres mots.

4. Indépendance de position

    Contrairement au RNN, ce calcul se fait en parallèle. Le mot 1 et le mot 100 sont traités à la même vitesse. L'attention connecte instantanément des mots très éloignés.

---