import spacy

nlp = spacy.load("en_core_web_sm")

text = "Autonomous cars shift insurance liability toward manufacturers"
doc = nlp(text)

print(f"{'MOT':<15} {'RELATION':<12} {'TÊTE (HEAD)':<15}")
print("-" * 45)

for token in doc:
    # token.text : le mot
    # token.dep_ : la relation de dépendance (ex: nsubj, dobj)
    # token.head.text : le mot auquel il est rattaché
    print(f"{token.text:<15} {token.dep_:<12} {token.head.text:<15}")

