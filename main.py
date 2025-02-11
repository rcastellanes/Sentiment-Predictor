import pandas as pd
from src.data_collection import fetch_tweets, fetch_tsla_data
from src.sentiment_analysis import analyze_sentiment
from src.model import build_and_train_model

def main():
    # Define the analysis period (adjust dates as needed)
    start_date = '2023-12-01'
    end_date = '2023-12-31'
    
    # Step 1: Collect tweets data for TSLA
    print("Fetching tweets...")
    tweets_df = fetch_tweets(query="TSLA", since=start_date, until=end_date, max_tweets=1000)
    tweets_df.to_csv("data/tweets.csv", index=False)
    
    # Step 2: Analyze sentiment for each tweet
    print("Analyzing tweet sentiments...")
    tweets_df['sentiment'] = tweets_df['content'].apply(analyze_sentiment)
    tweets_df['date'] = pd.to_datetime(tweets_df['date']).dt.date
    daily_sentiment = tweets_df.groupby('date')['sentiment'].mean().reset_index()
    
    # Step 3: Fetch TSLA stock price data
    print("Fetching TSLA stock data...")
    tsla_df = fetch_tsla_data(start_date, end_date)
    tsla_df['Date'] = pd.to_datetime(tsla_df['Date']).dt.date
    
    # Merge daily sentiment with TSLA stock data on the date
    merged_df = pd.merge(tsla_df, daily_sentiment, left_on='Date', right_on='date', how='inner')
    
    # Create target: next day's closing price (shift -1)
    merged_df = merged_df.sort_values('Date')
    merged_df['target'] = merged_df['Close'].shift(-1)
    merged_df = merged_df.dropna(subset=['target'])
    
    # Optional: Save merged data
    merged_df.to_csv("data/merged_data.csv", index=False)
    
    # Step 4: Build, train, and evaluate the prediction model
    print("Training prediction model...")
    build_and_train_model(merged_df)

if __name__ == "__main__":
    main()
