import pandas as pd
import yfinance as yf
import requests
import concurrent.futures
from datetime import timedelta

def get_news(api_key, query, from_date, page=1, page_size=100, max_retries=5):
    """
    Retrieve news articles from NewsAPI for a given query and date range,
    sorted by popularity. Only the first 100 results (page 1) are returned.
    Implements exponential backoff on 429 errors.
    """
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'language': 'en',
        'sortBy': 'popularity',  # Return the most popular articles
        'pageSize': page_size,
        'page': page,
        'apiKey': api_key
    }
    
    retry_count = 0
    while retry_count < max_retries:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        elif response.status_code == 429:
            # Too many requests: implement exponential backoff
            wait_time = 2 ** retry_count
            print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            retry_count += 1
        else:
            print(f"Error {response.status_code}: {response.text}")
            break
    return []

def collect_news_data(api_key, query, start_date, end_date, page_size=100):
    """
    Collect news articles day by day between start_date and end_date.
    Returns a DataFrame with the publication date and article content.
    Only collects the first 100 popular articles per day.
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    data = []
    
    for date in date_range:
        from_date = date.strftime("%Y-%m-%d")
        
        print(f"Collecting news for {from_date}...")
        articles = get_news(api_key, query, from_date, page=1, page_size=page_size)
        for article in articles:
            # Prefer the 'content'; if not available, use the 'description'
            content = article.get('content') or article.get('description', '')
            data.append({'date': from_date, 'content': content})
    
    df = pd.DataFrame(data)
    return df

def get_stock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches AMC historical stock data using yfinance.
    
    :param start_date: Start date in YYYY-MM-DD format.
    :param end_date: End date in YYYY-MM-DD format.
    :return: DataFrame with stock data.
    """

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
    session.mount('https://', adapter)

    tsla = yf.Ticker("AMC", session=session)
    tsla_df = tsla.history(start=start_date, end=end_date)
    tsla_df.reset_index(inplace=True)
    return tsla_df