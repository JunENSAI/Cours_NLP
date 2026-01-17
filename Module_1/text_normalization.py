import re
from nltk.corpus import stopwords

# Texte brut complexe
input_text = "      IN 2025, following the massive DATA BREACH disclosed on 07/18/2025, investigators revealed that over 1.500.000 records—including user_IDs (e.g., U-99821#A), hashed passwords (SHA256$9f2@!), phone numbers, and transactions totaling €12,008—were exfiltrated via an unsecured API endpoint (/v1/export?debug=true), triggering alerts at 02:14:33 UTC, regulatory fines under GDPR_ART.33, internal emails marked “URGENT!!!”, incident_refs=[INC-2025-LEAK-Ω], and a public statement asserting “NO SYSTEM IS 100% IMMUNE,” punctuated by logs showing latency=1.42s, retries=5/10, and access from IPs like 185.203.44.7 before containment  ."

stop_words = set(stopwords.words('english'))

# 1. Tout en minuscule
text_processed = input_text.lower()

# 2. Enlever les chiffres
text_processed = re.sub(r'\d+', '', text_processed)

# 3. Enlever la ponctuation (garde mots et espaces)
text_processed = re.sub(r'[^\w\s]', '', text_processed)

# 4. Enlever les espaces superflus (début/fin)
text_processed = text_processed.strip()

# Conversion en liste pour filtrage
lst_input_text = text_processed.split()
print(f"Liste intermédiaire : {lst_input_text[:10]}...") # Affichage partiel

# 5. Enlever les stopwords et reconstruction
final_tokens = [word for word in lst_input_text if word not in stop_words]
final_text = " ".join(final_tokens)

print("\n--- Texte Final Normalisé ---")
print(final_text)