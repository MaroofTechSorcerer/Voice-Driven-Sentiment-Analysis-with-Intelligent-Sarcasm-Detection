import joblib
from backend.utils.preprocess import preprocess_text

# Load the saved sarcasm detection model
model = joblib.load("backend/sarcasm/sarcasm_model.pkl")
vectorizer = joblib.load("backend/sarcasm/sarcasm_vectorizer.pkl")

def detect_sarcasm(text):
    """
    Predict if the given text is sarcastic or not.
    Returns True (sarcasm) or False (not sarcasm).
    """
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)

    return bool(prediction[0])  # True if sarcastic
