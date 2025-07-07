from textblob import TextBlob

def classify_sentiment(text):
    """
    Classifies the sentiment of the input text using TextBlob.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"
