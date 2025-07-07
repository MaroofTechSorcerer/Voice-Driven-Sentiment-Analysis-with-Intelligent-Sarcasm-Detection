import json
import os
import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# 1. Load Custom MUStARD JSON Format
# -----------------------------
with open("data/MUStARD/data/sarcasm_data.json", "r") as f:
    data = json.load(f)

rows = []
for key, value in data.items():
    text = value['utterance']
    label = 1 if value['sarcasm'] else 0
    rows.append((text, label))

df = pd.DataFrame(rows, columns=['text', 'label'])
print("Dataset loaded:", df.shape)

# -----------------------------
# 2. Preprocessing function
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

df['cleaned_text'] = df['text'].apply(preprocess)

# -----------------------------
# 3. Train/Test split
# -----------------------------
X = df['cleaned_text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train the model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# 5. Save the model and vectorizer
# -----------------------------
os.makedirs("backend/sarcasm", exist_ok=True)

joblib.dump(model, "backend/sarcasm/sarcasm_model.pkl")
joblib.dump(vectorizer, "backend/sarcasm/sarcasm_vectorizer.pkl")

print("Model and vectorizer saved!")
