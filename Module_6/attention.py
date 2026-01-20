import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes", split="train")

sample_text = dataset[15]['text']
print(f"Phrase sélectionnée : '{sample_text}'")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

inputs = tokenizer(sample_text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"\nTokens : {tokens}")

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

attention_matrix = outputs.attentions[-1] 


att_head = attention_matrix[0, 3, :, :]

print(f"\nShape de la matrice d'attention : {att_head.shape}")


plt.figure(figsize=(10, 8))
sns.heatmap(att_head, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)

plt.title("Visualisation du Self-Attention (Qui regarde Qui ?)")
plt.xlabel("Key (Ce qui est regardé)")
plt.ylabel("Query (Le mot qui regarde)")
plt.show()
