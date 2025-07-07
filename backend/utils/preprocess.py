import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download once
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Clean and preprocess input text:
    - Lowercase
    - Remove punctuation
    - Tokenize
    - Remove stopwords
    - Return cleaned text string
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Re-join tokens
    cleaned_text = " ".join(filtered_tokens)
    
    return cleaned_text
