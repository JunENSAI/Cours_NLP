# Cours_NLP

Le NLP (Natural Language Processing) est une branche de l’intelligence artificielle qui permet aux machines d’analyser, comprendre et générer le langage humain.

Pour prendre en main le repo voici quelques étapes qui pourraient être utiles :

- Créer un environnement virtuel pour ne pas avoir de problème de dépendances

```bash
python3 -m venv nlp_env
```
- Activer l'environnement virtuel que vous venez de créer :

```bash
source ./Cours_nlp/nlp_env/bin/activate
```

- Installez les librairies nécessaires pour le bon fonctionnement des fichiers python et jupyter notebook

```bash
cd Cours_NLP
pip install -r requirements.txt
```

- Executez en premier lieu le fichier `config_download.py` pour télécharger tout ce qui est nécessaire pour le bon fonctionnement de `nltk`

```bash
python3 config_download.py
```

- Puis, executez la commande suivante pour télécharger le modèle de spacy (`en_core_web_sm`) [sm : small]

```bash
python3 -m spacy download en_core_web_sm
```
