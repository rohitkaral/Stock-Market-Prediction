import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

def fetch_stock_data(ticker, start_date, end_date):
    """
    Retrieves historical stock price data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
        start_date (str): The start date for the data (e.g., '2020-01-01').
        end_date (str): The end date for the data (e.g., '2023-01-01').

    Returns:
        pandas.DataFrame: A DataFrame containing the stock price data, or None if an error occurs.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data available for ticker {ticker} between {start_date} and {end_date}.")
            return None
        return stock_data
    except Exception as error:
        print(f"Problem encountered while downloading data for {ticker}: {error}")
        return None

def prepare_data_for_model(stock_data_df, prediction_date=None):
    """
    Prepares the stock price data for modeling by creating features and
    defining the target variable.

    Args:
        stock_data_df (pandas.DataFrame): The input DataFrame containing stock price data.
        prediction_date (str, optional): The future date for which to predict the closing price
            (e.g., '2023-01-15'). If None, predicts 5 days in the future.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame, or None if the input is invalid.
    """
    if stock_data_df is None or stock_data_df.empty:
        print("Error: Empty DataFrame provided for preprocessing.")
        return None

    df = stock_data_df.copy()
    # Feature Engineering: Lagged Prices - using past closing prices as features
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)

    # Target Variable: Future Price - what we want to predict
    if prediction_date:
        try:
            target_date = pd.to_datetime(prediction_date)
            latest_available_date = df.index[-1]
            if target_date <= latest_available_date:
                print(f"Prediction date {prediction_date} is not in the future.  Please provide a future date.")
                return None
            days_to_predict = (target_date - latest_available_date).days
            if days_to_predict <= 0:
                print("Error: Prediction date must be after the last available date in the data.")
                return None
            df['Future_Close'] = df['Close'].shift(-days_to_predict)
        except ValueError:
            print(
                "Invalid prediction date format. Please use %Y-%m-%d format.  "
                "Using default 5-day prediction."
            )
            df['Future_Close'] = df['Close'].shift(-5)
    else:
        df['Future_Close'] = df['Close'].shift(-5)  # Default prediction target

    # Calculate daily returns - useful for understanding price changes
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate moving averages - to smooth out price fluctuations
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()

    # Calculate the Relative Strength Index (RSI) - a momentum indicator
    price_diff = df['Close'].diff()
    up_prices = price_diff.copy()
    down_prices = price_diff.copy()
    up_prices[up_prices < 0] = 0
    down_prices[down_prices > 0] = 0
    roll_up1 = up_prices.ewm(span=14).mean()
    roll_down1 = down_prices.abs().ewm(span=14).mean()
    relative_strength = roll_up1 / roll_down1
    df['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))

    df = df.dropna()  # Remove rows with missing values
    df = df.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
    return df

def build_model(X_train, y_train):
    """
    Builds and trains a Random Forest Regressor model.

    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The training target variable.

    Returns:
        sklearn.ensemble.RandomForestRegressor: The trained Random Forest Regressor model,
        or None if an error occurs during training.
    """
    try:
        # Initialize and train the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune hyperparameters here
        model.fit(X_train, y_train)
        return model
    except Exception as error:
        print(f"Error encountered while training the model: {error}")
        return None

def assess_model_performance(model, X_test, y_test, prediction_date=None):
    """
    Evaluates the trained model's performance and prints relevant metrics.

    Args:
        model (sklearn.ensemble.RandomForestRegressor): The trained model.
        X_test (pandas.DataFrame): The test features.
        y_test (pandas.Series): The test target variable.
        prediction_date (str, optional): The future date for which the prediction was made.

    Returns:
        numpy.ndarray: The predictions. Returns None on error.
    """
    if model is None:
        print("Error: Model is None in assess_model_performance.")
        return None
    try:
        # Generate predictions using the trained model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")
        return predictions
    except Exception as error:
        print(f"Error encountered while evaluating the model: {error}")
        return None

def visualize_predictions(y_test, predictions, ticker, prediction_date=None):
    """
    Visualizes the actual and predicted future closing prices.

    Args:
        y_test (pandas.Series): The actual future closing prices.
        predictions (numpy.ndarray): The predicted future closing prices.
        ticker (str): The stock ticker.
        prediction_date (str, optional): The future date for which the prediction was made.
    """
    if predictions is None:
        print("No predictions available to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Future Close Price', color='blue')
    plt.plot(y_test.index, predictions, label='Predicted Future Close Price', color='red')
    plot_title = f'{ticker} Stock Price Prediction'
    if prediction_date:
        plot_title += f' for {prediction_date}'
    plt.title(plot_title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_stock_prediction(ticker='AAPL', start_date='2020-01-01', end_date=None, prediction_date=None):
    """
    Orchestrates the stock price prediction process.

    Args:
        ticker (str): The stock ticker symbol.  Defaults to Apple (AAPL).
        start_date (str): The start date for the data. Defaults to '2020-01-01'.
        end_date (str, optional): The end date for the data.  Defaults to today's date.
        prediction_date (str, optional): The future date to predict. Defaults to None (5 days).
    """
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    try:
        stock_data_df = fetch_stock_data(ticker, start_date, end_date)
        if stock_data_df is None:
            print("Failed to download stock data. Exiting.")
            return

        processed_data_df = prepare_data_for_model(stock_data_df, prediction_date)
        if processed_data_df is None:
            print("Failed to preprocess data. Exiting.")
            return

        # Prepare data for modeling
        features = [col for col in processed_data_df.columns if col not in ['Future_Close']]
        target = 'Future_Close'

        X = processed_data_df[features]
        y = processed_data_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if the training set is empty
        if X_train.empty:
            print(
                "Error: The training set is empty.  This is likely because the start date is too close to the end date, "
                "or the end date is before any data."
            )
            return

        model = build_model(X_train, y_train)
        if model is None:
            print("Failed to create model. Exiting.")
            return

        predictions = assess_model_performance(model, X_test, y_test, prediction_date)
        if predictions is not None:
            visualize_predictions(y_test, predictions, ticker, prediction_date)
    except Exception as error:
        print(f"An unexpected error occurred: {error}")

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper() or 'AAPL'
    start_date = input("Enter start date (YYYY-MM-DD): ")  # Removed default start date
    end_date = input("Enter end date (YYYY-MM-DD): ") or None
    prediction_date = input(
        "Enter future date to predict (YYYY-MM-DD, press Enter for default 5 days): "
    ) or None
    run_stock_prediction(ticker, start_date, end_date, prediction_date)
