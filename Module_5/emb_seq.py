import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# Padding

raw_sentences = [
    [1, 2, 3],
    [4, 5, 6, 7, 8],
    [9]
]

tensors = [torch.LongTensor(s) for s in raw_sentences]

padded_batch = pad_sequence(tensors, batch_first=True, padding_value=0)

print(f"Shape résultante : {padded_batch.shape}")
print("Matrice après padding :\n", padded_batch)

attention_mask = (padded_batch != 0).long()

print("Masque :\n", attention_mask)

lengths = torch.tensor([len(s) for s in raw_sentences])

packed_input = pack_padded_sequence(padded_batch, lengths, batch_first=True, enforce_sorted=False)

print("Packed Sequence (version optimisée) :")
print(packed_input.data) 

# Embedding layer

embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)

# Simuler une phrase : "Le(1) chat(4) dort(9)"
input_indices = torch.LongTensor([1, 4, 9])

# Passage dans la couche
output_vectors = embedding_layer(input_indices)

print(f"Indices d'entrée : {input_indices}")
print(f"Sortie (Shape {output_vectors.shape}) :")
print(output_vectors)

pretrained_weights = torch.FloatTensor([
    [0, 0, 0], # padding (index 0)
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4]
])

pretrained_layer = nn.Embedding.from_pretrained(pretrained_weights)

# Test avec le mot 2 et 4
input_test = torch.LongTensor([2, 4])
print("\n--- Expérience 2 : Poids Pré-entraînés ---")
print(f"Entrée : {input_test}")
print(f"Sortie :\n{pretrained_layer(input_test)}")

# Pour geler (empêcher l'apprentissage sur cette couche) :
pretrained_layer.weight.requires_grad = False

print("\n--- Expérience 3 : Vérification du Gel ---")
print(f"La couche apprend-elle ? -> {pretrained_layer.weight.requires_grad}")
# Si False, l'optimizer ignorera cette couche lors de la mise à jour des poids.
