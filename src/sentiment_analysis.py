from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon (only once)
nltk.download('vader_lexicon', quiet=True)

# Initialize the sentiment analyzer globally
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> float:
    """
    Analyze the sentiment of a text string using VADER.
    
    :param text: The text to analyze.
    :return: The compound sentiment score (range: -1 to 1).
    """
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores['compound']