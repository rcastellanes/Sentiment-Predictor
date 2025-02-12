# src/main.py

from datetime import datetime
from datetime import timedelta
from src.data_collection import collect_news_data, get_stock_data
from src.sentiment_analysis import apply_sentiment_to_df, aggregate_daily_sentiment
from src.model import prepare_stock_trend, merge_sentiment_stock, train_model
from dotenv import load_dotenv
import os

# Only load .env if NEWS_API_KEY isn't already set
if not os.getenv("NEWS_API_KEY"):
    load_dotenv()

news_api_key = os.getenv("NEWS_API_KEY")
if not news_api_key:
    raise ValueError("No API key found. Please set the NEWS_API_KEY environment variable.")

def main():
    # Replace with your NewsAPI key
    news_api_key = "45a8fb7524ce41f2803f21b771eb5617"
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    # Format the dates as strings in YYYY-MM-DD format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Query for AMC (limiting to the ticker)
    query = "AMC"
    
    # Step 1: Collect News Data for AMC
    print("Collecting news data...")
    news_df = collect_news_data(api_key=news_api_key, query=query, start_date=start_date, end_date=end_date, page_size=100)
    if news_df.empty:
        print("No news articles were fetched. Check your API key and query parameters.")
        return
    
    # Step 2: Apply Sentiment Analysis on News Articles
    print("Analyzing news sentiment...")
    news_df = apply_sentiment_to_df(news_df, text_column='content')
    daily_sentiment = aggregate_daily_sentiment(news_df)
    
    # Step 3: Get AMC Stock Data
    print("Collecting stock data for AMC...")
    stock_data = get_stock_data(start_date=start_date, end_date=end_date)
    stock_trend = prepare_stock_trend(stock_data)
    
    # Step 4: Merge News Sentiment and Stock Data
    merged_data = merge_sentiment_stock(daily_sentiment, stock_trend)
    if merged_data.empty:
        print("No merged data available. Please verify date ranges and data collection.")
        return
    
    # Step 5: Train and Evaluate the Model
    print("Training model...")
    model = train_model(merged_data)

if __name__ == "__main__":
    main()