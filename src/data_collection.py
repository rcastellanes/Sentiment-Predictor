import pandas as pd
import snscrape.modules.twitter as sntwitter
import yfinance as yf

def fetch_tweets(query: str, since: str, until: str, max_tweets: int = 1000) -> pd.DataFrame:
    """
    Fetch tweets containing the query string within a date range.
    
    :param query: Keyword to search for (e.g., "TSLA").
    :param since: Start date in YYYY-MM-DD format.
    :param until: End date in YYYY-MM-DD format.
    :param max_tweets: Maximum number of tweets to fetch.
    :return: DataFrame with tweet 'date' and 'content'.
    """
    tweets_list = []
    query_str = f'{query} since:{since} until:{until}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query_str).get_items()):
        if i >= max_tweets:
            break
        tweets_list.append({
            'date': tweet.date,
            'content': tweet.content
        })
    tweets_df = pd.DataFrame(tweets_list)
    return tweets_df

def fetch_tsla_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch TSLA historical stock data using yfinance.
    
    :param start_date: Start date in YYYY-MM-DD format.
    :param end_date: End date in YYYY-MM-DD format.
    :return: DataFrame with stock data.
    """
    tsla = yf.Ticker("TSLA")
    tsla_df = tsla.history(start=start_date, end=end_date)
    tsla_df.reset_index(inplace=True)
    return tsla_df