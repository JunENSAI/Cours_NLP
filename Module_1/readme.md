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

## Lemmatization et Stemming

### Lemmatization

La lemmatisation est une technique de normalisation de texte basée sur la linguistique qui convertit les mots en leur forme de base dans le dictionnaire, appelée lemme, en tenant compte de la grammaire, du vocabulaire et du contexte.

- Prend en compte la catégorie grammaticale (POS) du mot identifié

- Le mot est comparé à une base de données lexicale

- Des règles grammaticales sont appliquées

- Les formes fléchies sont mises en correspondance avec leur lemme de base

- Un mot valide du dictionnaire est renvoyé

- Prise en compte du contexte et précision

- Plus coûteux en termes de calcul


**Techniques utilisées**

- Recherche dans le dictionnaire : met en correspondance les mots avec des formes de base valides

- Marquage POS : détermine le rôle grammatical

- Analyse morphologique : gère la flexion et la dérivation

`Example`

| Original Word| Lemma    | POS      |
|:-------------| :-------:| --------:|
| running     | run   | Verb    |
| better     | good   | Adjective    |
| studies     | study   | Noun    |
| was     | be   | Verb    |

`Application`

- **Analyse des sentiments** : la lemmatisation préserve le sens des mots, ce qui permet une classification plus précise des sentiments.

- **Résumé de texte** : elle aide à identifier les concepts clés en réduisant les mots à leurs formes de base significatives.

- **Chatbots et IA conversationnelle** : elle garantit une compréhension correcte de l'intention de l'utilisateur en préservant l'intégrité sémantique.

- **Systèmes de réponse aux questions** : elle améliore la correspondance entre les questions et les réponses en utilisant des mots de base valides.

- **Modélisation de sujets** : elle produit des sujets cohérents en regroupant correctement les mots sémantiquement liés.

### Stemming

Le stemming est une technique de normalisation de texte basée sur des règles qui réduit les mots à leur forme racine en supprimant les préfixes ou les suffixes. La forme résultante, appelée radical, peut ne pas être un mot valide ou significatif dans la langue.

- Chaque mot est traité indépendamment, sans tenir compte du contexte.

- L'algorithme recherche les suffixes ou préfixes courants.

- Des règles prédéfinies sont appliquées pour supprimer ces affixes.

- La partie restante du mot est renvoyée comme radical.

En bref, le stemming effectue une troncature mécanique des mots.

**Techniques utilisées**

- Suppression des suffixes : supprime les terminaisons courantes telles que `-ing, -ed, -es`.

- Troncature basée sur des règles : applique des règles linguistiques fixes.

- Réduction agressive : raccourcit les mots pour une généralisation maximale.

`Exemple`

| Original Word| Stem   | 
|:-------------| -------:|
| running     | run   |
| smiling    | smile   |
| studies     | studi   |

`Application`

- **Moteurs de recherche** : le stemming améliore la correspondance des requêtes en traitant les différentes formes morphologiques d'un mot comme un seul et même terme, ce qui augmente le taux de rappel dans les résultats de recherche.

- **Systèmes de recherche d'informations** : il aide à retrouver des documents pertinents même lorsque les termes de la requête et ceux du document diffèrent sur le plan grammatical.

- **Indexation de documents** : il réduit le nombre de termes uniques stockés dans les index, améliorant ainsi l'efficacité du stockage et la vitesse de recherche.

- **Classification de textes** : il simplifie la représentation des caractéristiques en regroupant les formes de mots apparentées sous une seule racine.

- **Systèmes de correspondance de mots-clés** : le stemming permet une correspondance approximative des mots-clés dans les systèmes NLP basés sur des règles ou légers.






