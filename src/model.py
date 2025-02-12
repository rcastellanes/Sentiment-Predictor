import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def prepare_stock_trend(stock_data):
    """
    Create a 'trend' column: 1 if the stock closed higher than it opened, else 0.
    Format the date column to match the sentiment data.
    """
    stock_data['trend'] = (stock_data['Close'] > stock_data['Open']).astype(int)
    stock_data['date'] = stock_data['Date'].dt.strftime("%Y-%m-%d")
    return stock_data[['date', 'trend']]

def merge_sentiment_stock(daily_sentiment, stock_trend):
    """
    Merge daily sentiment with the stock trend data.
    """
    merged = pd.merge(daily_sentiment, stock_trend, on='date', how='inner')
    return merged

def train_model(merged_data):
    """
    Train a Logistic Regression model to predict stock trend from daily sentiment.
    """
    X = merged_data[['sentiment']]
    y = merged_data['trend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", acc)
    return model