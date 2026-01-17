from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# 1. Texte à analyser
text = "If the party was over and our time on Earth was through"

# 2. Tokenisation et PoS Tagging
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print(f"Tags initiaux : {tags}\n")

# 3. Définition d'une grammaire de "Chunking"
# NP (Noun Phrase) : Un déterminant optionnel (?), suivi d'adjectifs (*), suivi d'un nom (+)
grammar = "NP: {<DT>?<JJ>*<NN>}"

# 4. Création du Parser
chunk_parser = RegexpParser(grammar)

# 5. Exécution du Chunking
tree = chunk_parser.parse(tags)

print("Résultat du Chunking (Arbre sous forme texte) :")
print(tree)

tree.draw() # pour visualiser l'arbre graphiquement