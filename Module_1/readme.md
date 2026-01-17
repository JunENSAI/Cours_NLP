# Traitement de texte et Pre-traitement

## Tokenization

- Consiste à diviser une chaîne ou un texte en une liste d'unités plus petites appelées tokens.

- Utilise un tokeniseur pour segmenter les données non structurées et le texte en langage naturel en blocs d'informations distincts, en les traitant comme des éléments différents.

- Tokens : mots ou sous-mots dans le contexte du traitement du langage naturel. Exemple : un mot est un token dans une phrase, un caractère est un token dans un mot, etc.

- Application : multiples tâches de NLP, traitement de texte, modélisation linguistique et traduction automatique.

### Types de Tokenization

1. Tokenisation des mots

    La tokenisation des mots est la méthode la plus couramment utilisée pour diviser un texte en mots individuels. Elle fonctionne bien pour les langues dont les mots ont des limites claires, comme l'anglais.
    
    ```plaintext
    Entrée avant tokenization: ["NLP is about tokenization"]

    Sortie après tokenization: ["NLP", "is", "about", "tokenization"]
    ```

2. Tokenization des caractères

    Dans la tokenisation des caractères, les données textuelles sont divisées et converties en une séquence de caractères individuels. Cela est utile pour les tâches qui nécessitent une analyse détaillée, telles que la correction orthographique ou les tâches dont les limites ne sont pas clairement définies. Cela peut également être utile pour modéliser le langage au niveau des caractères.

    ```plaintext
    Entrée avant tokenization: ["I am you"]

    Sortie après tokenization: ["I", "a", "m", "y", "o", "u"]
    ```

3. Tokenization des phrases

La tokenization est aussi une technique commune utilisée pour faire une division de paragraphe ou un large ensemble de phrases en phrases separées comme tokens. Ceci est utile pour les tâches qui recquiert une analyse de phrases individuelles ou de pre-traitement.

```plaintext
Entrée avant tokenization: ["I am you. You are not me"]

Sortie après tokenization: ["I am you.", "You are not me"]

```

4. Tokenization N-gram

La tokenization N-gram sépare les mots en une taille fixée (`size = n`) de donnée.

```plaintext
Entrée avant tokenization: ["Natural Language Processing is fun"]

Output when tokenized by bigrams: [('Natural', 'Language'), ('Language', 'Processing'), ('Processing', 'is'), ('is', 'fun')]
```

### Utilisation de la Tokenization

La tokenisation est une étape essentielle du traitement de texte et du traitement du langage naturel (NLP) pour plusieurs raisons. Certaines d'entre elles sont énumérées ci-dessous :

- Traitement du texte : réduit la taille du texte brut, ce qui facilite et optimise l'analyse statistique et computationnelle.

- Extraction de caractéristiques : les données textuelles peuvent être représentées numériquement pour la compréhension algorithmique en utilisant des tokens comme caractéristiques dans les modèles ML.

- Récupération d'informations : la tokenisation est essentielle pour l'indexation et la recherche dans les systèmes qui stockent et récupèrent efficacement des informations sur la base de mots ou d'expressions.

- Analyse de texte : utilisée dans l'analyse des sentiments et la reconnaissance d'entités nommées, pour déterminer la fonction et le contexte de chaque mot dans une phrase.

- Gestion du vocabulaire : génère une liste de jetons distincts, aide à gérer le vocabulaire d'un corpus.
Adaptation à des tâches spécifiques : s'adapte aux besoins d'une tâche particulière de NLP, utile pour la synthèse et la traduction automatique.

