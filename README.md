# Sentiment-Based Stock Price Predictor

## Project Overview
This project aims to predict AMC (AMC) stock price movements by analyzing News Article sentiment. The pipeline consists of:
- **Data Collection:** Using NewsAPI to gather recent tweets containing "AMC".
- **Sentiment Analysis:** Analyzing each news article’s sentiment with NLTK’s VADER.
- **Stock Data Collection:** Fetching TSLA historical stock data via `yfinance`.
- **Feature Engineering & Modeling:** - Aggregating daily sentiment scores and using them—along with today’s closing price—as features to predict the next day’s increase via a logistic regression model.

tsla_sentiment_analysis/
├── README.md
├── requirements.txt
├── main.py                   # Main script that ties everything together
└── src/
    ├── data_collection.py        # Collects Twitter and TSLA stock data
    ├── sentiment_analysis.py     # Applies sentiment analysis to tweets
    ├── model.py                  # Prepares the dataset and trains a Logistic Regression model