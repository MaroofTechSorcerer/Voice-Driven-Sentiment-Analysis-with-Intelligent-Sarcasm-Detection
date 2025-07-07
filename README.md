# 🎙️ Voice-Driven Sentiment Analysis with Intelligent Sarcasm Detection

This project analyzes spoken audio to detect the underlying **sentiment** and **sarcasm** in human speech. It's built using Python, Streamlit, and ML/NLP tools like TextBlob and the MUStARD dataset.

---

## 🚀 Features
- ✅ Convert voice to text (speech-to-text)
- ✅ Preprocess and clean text
- ✅ Detect sentiment (Positive / Negative / Neutral)
- ✅ Detect sarcasm using a custom-trained model (MUStARD)
- ✅ Streamlit-based web interface
- ✅ Beginner-friendly code and project structure

---

## 🧱 Tech Stack
- Python
- Streamlit (Frontend)
- SpeechRecognition (Voice Input)
- TextBlob (Sentiment)
- Scikit-learn (Sarcasm model)
- MUStARD Dataset
- NLTK, TF-IDF

---

## 📁 Folder Structure


voice-sentiment-sarcasm/
├── app/
│   └── streamlit_app.py
├── backend/
│   ├── audio/
│   │   └── speech_to_text.py
│   ├── sentiment/
│   │   └── sentiment_classifier.py
│   ├── sarcasm/
│   │   └── sarcasm_detector.py
│   │   └── sarcasm_model.pkl
│   │   └── sarcasm_vectorizer.pkl
│   ├── utils/
│   │   └── preprocess.py
├── data/
│   └── mustard/ (cloned from MUStARD GitHub) [MUStARD dataset]
├── notebooks/
│   └── train_sarcasm_model.ipynb   (Sarcasm model training notebook)
├── outputs/    (Results and logs)
├── main.py     (Main runner for backend)
├── README.md   
├── requirements.txt    (Python dependencies)


