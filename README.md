# Sentiment-Based Stock Price Predictor
The overall challenge is to see whether social media and news activity have an impact on stock price. To limit the scope, I included only one stock, AMC, which has been historically news-driven and limited the scope to the past 30 days due to API limits. For this exercise, I only limited to news articles.

## Project Overview
This project aims to predict AMC (AMC) stock price movements by analyzing News Article sentiment. The pipeline consists of:
- **Data Collection:** Using NewsAPI to gather recent tweets containing "AMC".
- **Sentiment Analysis:** Analyzing each news article’s sentiment with NLTK’s VADER.
- **Stock Data Collection:** Fetching TSLA historical stock data via `yfinance`.
- **Feature Engineering & Modeling:** - Aggregating daily sentiment scores and using them—along with today’s closing price—as features to predict the next day’s increase via a logistic regression model. I wanted to use a simple classification model to determine whether or not I have a viable product.
  
## Challenges Faced
- **Data Noise:** News articles were limited to the main content of the article. I also chose the top 100 articles by popularity defined by NewsAPI, which may have a natural bias with content.
- **API Rate Limits:** The NewsAPI limits the number of requests; thus, data collection must handle rate limits accordingly. I also used a free 
- **Temporal Alignment:** Matching the timing of news articles with market trading hours required careful aggregation.
- **Feature Engineering:** Aggregating sentiment into meaningful predictors for stock movements is an iterative process.

## Results Interpretation
- The model prints an accuracy score on a held-out test set. Accuracy above 50% may suggest some predictive signal, though additional features and models are likely needed for robust performance. For this case, I saw accuracy scores at 70% and above for the test set. This indicates that there is a positive relationship between sentiment and stock price.
- The aggregated sentiment is a crude feature and future work might include volume of news articles, source influence, or additional market indicators.

## Next Steps
- **Data Enhancement:** Integrate other sources (e.g., X tweets, Reddit posts) to improve the feature set.
- **Expansion of Data:** Include other stocks as well to determine whether or not the model can be expanded to other stocks.
- **More Advanced Models:** Experiment with more advanced algorithms such as ensemble methods or neural networks.
- **Real-Time Prediction:** Build a pipeline for real-time sentiment analysis and live predictions.
- **Feature Engineering:** Explore additional features (news volume, news shars, etc.) to enhance predictive power.

## Deployment Considerations
- **Scalability:** Deploy on cloud platforms (AWS, GCP) for scalable data ingestion and model serving.
- **Monitoring:** Implement monitoring to track model performance over time.
- **API Management:** Use caching or batching to manage API rate limits.
- **Compliance:** Ensure compliance with data privacy and API usage policies.


```bash
Sentiment-Predictor/
├── README.md
├── requirements.txt
├── main.py                   # Main script that ties everything together
└── src/
    ├── data_collection.py        # Collects AMC News and AMC stock data
    ├── sentiment_analysis.py     # Applies sentiment analysis to tweets
    ├── model.py                  # Prepares the dataset and trains a Logistic Regression model
```
