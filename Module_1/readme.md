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