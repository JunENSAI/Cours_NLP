import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

input_text = "      IN 2025, following the massive DATA BREACH disclosed on 07/18/2025, investigators revealed that over 1.500.000 records—including user_IDs (e.g., U-99821#A), hashed passwords (SHA256$9f2@!), phone numbers, and transactions totaling €12,008—were exfiltrated via an unsecured API endpoint (/v1/export?debug=true), triggering alerts at 02:14:33 UTC, regulatory fines under GDPR_ART.33, internal emails marked “URGENT!!!”, incident_refs=[INC-2025-LEAK-Ω], and a public statement asserting “NO SYSTEM IS 100% IMMUNE,” punctuated by logs showing latency=1.42s, retries=5/10, and access from IPs like 185.203.44.7 before containment  ."

# tout en minuscule
lower_input_text = input_text.lower()

# enleve les chiffres
no_number_input_text = re.sub(r'\d+','',lower_input_text)

# enlève les ponctuations
no_punc_input_text = re.sub(r'[^\w\s]','', no_number_input_text) 

# enlève les espaces
no_wspace_input_text = no_punc_input_text.strip()
no_wspace_input_text

# conversion de l'entree en liste de mots
lst_input_text = [no_wspace_input_text][0].split()
print(lst_input_text)

# enlève les stopwords
no_stpwords_input_text=""
for i in lst_input_text:
    if not i in stop_words:
        no_stpwords_input_text += i+' '
        
# enlève le dernier espacement
no_stpwords_input_text = no_stpwords_input_text[:-1]

print(no_stpwords_input_text)