import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)

print(f"{'ENTITÉ':<20} {'LABEL':<10} {'EXPLICATION'}")
print("-" * 50)

for ent in doc.ents:
    # ent.text : le texte de l'entité
    # ent.label_ : l'étiquette (ORG, GPE, MONEY...)
    # spacy.explain() : donne la définition de l'étiquette
    print(f"{ent.text:<20} {ent.label_:<10} {spacy.explain(ent.label_)}")