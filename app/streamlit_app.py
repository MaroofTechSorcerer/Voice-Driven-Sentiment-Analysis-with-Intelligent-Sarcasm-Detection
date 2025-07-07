# frontend for voice-based sentiment analysis and sarcasm detection
import streamlit as st
from backend.audio.speech_to_text import speech_to_text
from backend.utils.preprocess import preprocess_text
from backend.sentiment.sentiment_classifier import classify_sentiment
from backend.sarcasm.sarcasm_detector import detect_sarcasm

st.set_page_config(page_title="Voice-Based Sentiment & Sarcasm Detector", layout="centered")
st.title("🎙️ Voice-Driven Sentiment Analysis with Sarcasm Detection")

st.markdown("Upload an audio file or record your voice to analyze sentiment and detect sarcasm.")

# --- Upload Section ---
audio_file = st.file_uploader("Upload a .wav file", type=["wav"])

if st.button("Analyze"):
    if audio_file:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.read())

        # Step 1: Speech to Text
        text = speech_to_text("uploaded_audio.wav")
        st.markdown(f"### 📝 Transcribed Text:\n{text}")

        # Step 2: Preprocess
        cleaned = preprocess_text(text)
        st.markdown(f"### 🧼 Cleaned Text:\n{cleaned}")

        # Step 3: Sentiment
        sentiment = classify_sentiment(cleaned)
        st.success(f"💬 Sentiment: **{sentiment}**")

        # Step 4: Sarcasm Detection
        sarcasm = detect_sarcasm(text)
        if sarcasm:
            st.warning("🙃 Sarcasm Detected!")
        else:
            st.info("🙂 No Sarcasm Detected.")

    else:
        st.error("Please upload an audio file first.")
