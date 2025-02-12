import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon if not already available
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(text):
    """
    Return the compound sentiment score for a given text.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def apply_sentiment_to_df(df, text_column='content'):
    """
    Apply sentiment analysis to a DataFrame column containing article content.
    """
    df['sentiment'] = df[text_column].apply(analyze_sentiment)
    return df

def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment scores by day (mean compound score).
    """
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()
    return daily_sentiment