# TSLA Sentiment-Based Stock Price Predictor

## Project Overview
This project aims to predict Tesla (TSLA) stock price movements by analyzing Twitter/X sentiment. The pipeline consists of:
- **Data Collection:** Using `snscrape` to gather recent tweets containing "TSLA".
- **Sentiment Analysis:** Analyzing each tweet’s sentiment with NLTK’s VADER.
- **Stock Data Collection:** Fetching TSLA historical stock data via `yfinance`.
- **Feature Engineering & Modeling:** - Aggregating daily sentiment scores and using them—along with today’s closing price—as features to predict the next day’s closing price via a linear regression model.