import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def build_and_train_model(data):
    """
    Build, train, and evaluate a linear regression model using aggregated daily sentiment
    and today's closing price to predict the next day's closing price.

    :param data: DataFrame containing at least 'sentiment', 'Close', and 'target' columns.
    """
    # Features: aggregated sentiment and today's closing price
    X = data[['sentiment', 'Close']]
    # Target: next day's closing price
    y = data['target']

    # Split data into training and testing sets (without shuffling to preserve time order)
    split_index = int(0.8 * len(data))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot actual vs. predicted closing prices
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.reset_index(drop=True), label="Actual", marker='o')
    plt.plot(y_pred, label="Predicted", marker='x')
    plt.legend()
    plt.title("Predicted vs Actual TSLA Next Day Closing Prices")
    plt.xlabel("Sample (Ordered in Time)")
    plt.ylabel("TSLA Closing Price")
    plt.show()
