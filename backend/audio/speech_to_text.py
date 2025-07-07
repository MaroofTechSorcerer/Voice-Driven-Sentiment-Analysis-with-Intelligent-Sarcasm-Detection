import speech_recognition as sr

def speech_to_text(audio_path=None):
    """
    Converts speech to text from a file or microphone.
    If audio_path is None, records from microphone.
    Returns the transcribed text.
    """
    recognizer = sr.Recognizer()

    try:
        if audio_path:
            # Use pre-recorded audio file
            with sr.AudioFile(audio_path) as source:
                print("Reading audio file...")
                audio_data = recognizer.record(source)
        else:
            # Use microphone input
            with sr.Microphone() as source:
                print("Adjusting for ambient noise... Speak now.")
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source)

        print("Transcribing...")
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
        return text

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        return ""
    except sr.RequestError:
        print("API unavailable or limit reached.")
        return ""
