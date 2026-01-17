from nltk.tokenize import word_tokenize, sent_tokenize

# word tokenization

t = "I am back on the track."

token_words = word_tokenize(t)

print(f"résultat de la tokenization par mots : {token_words}")

# sentence tokenization

s = "I am you. You are not me."

token_sentences = sent_tokenize(s)

print(f"résultat de la tokenization par phrases : {token_sentences}")