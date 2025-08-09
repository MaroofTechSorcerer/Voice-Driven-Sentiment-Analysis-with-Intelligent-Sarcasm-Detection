# 🎙️ Voice-Driven-Sentiment-Analysis-with-Intelligent-Sarcasm-Detection : Hear the Unspoken 😒

**Decoding the true sentiment and subtle sarcasm hidden in human speech.**

This Project goes beyond literal transcription. It's an AI-powered tool that listens to what you say and understands *how* you say it, distinguishing genuine praise from witty sarcasm.
---

## 🎯 Core Features

-   🎤 **Real-time Voice Transcription**: Instantly convert spoken words into text using your microphone.
-   📊 **Precision Sentiment Analysis**: Classify text as `Positive`, `Negative`, or `Neutral` with high accuracy.
-   💡 **Intelligent Sarcasm Detection**: Our custom-trained model, built on the `MUStARD` dataset, identifies sarcasm that other tools miss.
-   🧠 **Context-Aware Engine**: Understands that "Yeah, *great*..." can mean two very different things based on context.
-   ✨ **Interactive Web Interface**: A clean and user-friendly dashboard built with Streamlit.

---

## ⚙️ How It Works

The magic happens in a simple, powerful pipeline:

1.  **Capture**: Your voice is captured via the browser and sent to our Python backend.
2.  **Transcribe**: `SpeechRecognition` converts the audio into raw text.
3.  **Clean**: The text is preprocessed using `NLTK` (tokenization, stop-word removal).
4.  **Analyze**:
    * **Sentiment**: `TextBlob` provides an initial sentiment polarity score.
    * **Sarcasm**: Our pre-trained `Scikit-learn` model (using a `TF-IDF` vectorizer) analyzes the text for sarcastic cues.
5.  **Visualize**: The final verdict is beautifully displayed in the Streamlit interface.

---

## 🛠️ Tech Stack

-   **Frontend**: `Streamlit`
-   **Backend**: `Python`
-   **Speech-to-Text**: `SpeechRecognition` & `PyAudio`
-   **NLP & Modeling**: `scikit-learn`, `TextBlob`, `NLTK`, `Pandas`
-   **Core Dataset**: `MUStARD` (Multimodal Sarcasm Detection Dataset)

---

## 🚀 Get Started in 3 Steps

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/voice-sentiment-sarcasm.git](https://github.com/your-username/voice-sentiment-sarcasm.git)
cd voice-sentiment-sarcasm
````

**2. Install Dependencies**

```bash
# It's highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
python -m nltk.downloader punkt
```

*Note: You may need to install system-level libraries for `PyAudio` to work.*

  - *On Debian/Ubuntu: `sudo apt-get install portaudio19-dev`*
  - *On macOS: `brew install portaudio`*

**3. Run the App\!**

```bash
streamlit run app/streamlit_app.py
```

Your browser should open with the running application.

-----

## 📁 Project Structure

The repository is organized for clarity and scalability:

```
voice-sentiment-sarcasm/
├── app/
│   └── streamlit_app.py         # The Streamlit frontend code
├── backend/
│   ├── audio/                   # Speech-to-text module
│   ├── sentiment/               # Sentiment analysis logic
│   ├── sarcasm/                 # Sarcasm detection model & logic
│   │   ├── sarcasm_model.pkl    # Pre-trained sarcasm classifier
│   │   └── sarcasm_vectorizer.pkl # Pre-trained TF-IDF vectorizer
│   └── utils/                   # Text preprocessing utilities
├── data/
│   └── mustard/                 # MUStARD dataset (download required)
├── notebooks/
│   └── train_sarcasm_model.ipynb  # Jupyter Notebook for model training
├── assets/
│   └── demo.gif                 # Example location for your demo GIF
├── main.py                      # Main backend runner (for non-Streamlit use)
├── README.md                    # You are here!
└── requirements.txt             # Project dependencies
```

-----

## 🧠 The Sarcasm Engine

Standard sentiment analysis tools often fail with sarcasm because they take text at face value. Our model was specifically trained on the **MUStARD dataset**, which includes text with contextual information from TV shows, providing rich examples of real-world sarcasm.

See the `notebooks/train_sarcasm_model.ipynb` to understand how the model was built from scratch\!

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/your-username/voice-sentiment-sarcasm/issues).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----
