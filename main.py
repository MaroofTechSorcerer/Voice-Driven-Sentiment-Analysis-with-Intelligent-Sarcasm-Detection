# Import necessary libraries
import os
from backend.audio.speech_to_text import speech_to_text
from backend.sentiment.sentiment_classifier import classify_sentiment
from backend.sarcasm.sarcasm_detector import detect_sarcasm
from backend.utils.preprocess import preprocess_text


def main():
    print("Welcome to the Voice-Driven Sentiment Analysis with Sarcasm Detection System!")
    
    # 1. Ask user to choose input method
    print("Choose input method:\n1. Microphone\n2. Audio File")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        text = speech_to_text()  # Mic input
    elif choice == "2":
        audio_input_path = "input_audio.wav"
        text = speech_to_text(audio_input_path)  # Use audio file
    else:
        print("Invalid choice.")
        return
    
    if not text:
        print("No text captured. Exiting.")
        return
    
    # 2. Preprocess the text (cleaning, tokenization, etc.)
    print("Preprocessing the transcribed text...")
    processed_text = preprocess_text(text)
    
    # 3. Sentiment Analysis
    print("Analyzing sentiment...")
    sentiment = classify_sentiment(processed_text)
    print(f"Sentiment: {sentiment}")
    
    # 4. Sarcasm Detection
    print("Detecting sarcasm...")
    sarcasm = detect_sarcasm(processed_text)
    print(f"Sarcasm detected: {sarcasm}")
    
    # 5. Final Output
    print("Generating final output...")
    if sarcasm:
        if sentiment == "Positive":
            sentiment = "Negative due to sarcasm"
        elif sentiment == "Negative":
            sentiment = "Positive due to sarcasm"
    
    print(f"Final sentiment after sarcasm adjustment: {sentiment}")

if __name__ == "__main__":
    main()
