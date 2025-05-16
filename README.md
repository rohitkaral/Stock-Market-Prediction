# Stock Market Price Prediction with Random Forest Regression

## Overview

This Python script uses the Random Forest Regression algorithm to predict future stock prices. It fetches historical stock data from Yahoo Finance, preprocesses the data, trains a model, and makes predictions.

**Disclaimer:** This code is for educational purposes only.  Stock market predictions are inherently uncertain, and this script should not be used for financial decision-making.  Past performance is not indicative of future results. Consult with a qualified financial advisor before making any investment decisions.

## Features

* Data Acquisition: Downloads historical stock price data from Yahoo Finance using the `yfinance` library.
* Feature Engineering:
    * Lagged closing prices (past 5 days)
    * Daily returns
    * 7-day and 30-day moving averages
    * Relative Strength Index (RSI)
* Model: Random Forest Regressor from scikit-learn.
* Prediction: Predicts the closing price of a stock for a specified future date or a default of 5 days into the future.
* Evaluation: Calculates and prints the Mean Squared Error (MSE) and R-squared (R2) to assess model performance.
* Visualization: Plots the actual and predicted future closing prices using `matplotlib`.
* Error Handling: Includes checks for invalid date ranges, empty data, and other potential issues.

## Prerequisites

Before running the script, ensure you have the following installed:

* Python: (Version 3.6 or later is recommended)
* Libraries: Install the required Python libraries using pip:

    ```bash
    pip install numpy pandas scikit-learn yfinance matplotlib
    ```

## Installation

1.  **Clone the repository** (or download the script) to your local machine:

    ```bash
    git clone <your_repository_url>
    cd <repository_directory>
    ```

2.  **Install the required libraries** (see "Prerequisites" above).

## Usage

1.  **Run the script:**

    ```bash
    python your_script_name.py  # Replace your_script_name.py
    ```

2.  **Enter the required information:**

    * Stock Ticker Symbol: Enter the ticker symbol of the stock you want to predict (e.g., AAPL for Apple, MSFT for Microsoft, GOOG for Google).
    * Start Date: Enter the date from which you want to begin collecting historical data (format: YYYY-MM-DD, e.g., 2020-01-01). This date should be in the past.
    * End Date: Enter the date until which you want to collect historical data (format: YYYY-MM-DD, e.g., 2023-12-31).
        * To use today's date, just press Enter.
    * Future Date to Predict: Enter the date for which you want to predict the closing price (format: YYYY-MM-DD, e.g., 2024-01-15). This date should be in the future.
        * To predict 5 days in the future, just press Enter.

## Example

    Enter stock ticker symbol (e.g., AAPL): AAPL
    Enter start date (YYYY-MM-DD): 2023-01-01
    Enter end date (YYYY-MM-DD): 2024-01-01
    Enter future date to predict (YYYY-MM-DD, press Enter for default 5 days): 2024-01-15

## Code Structure

* `fetch_stock_data(ticker, start_date, end_date)`: Downloads historical stock price data from Yahoo Finance.
* `prepare_data_for_model(stock_data_df, prediction_date=None)`: Preprocesses the data, engineers features, and defines the target variable.
* `build_model(X_train, y_train)`: Creates and trains the Random Forest Regressor model.
* `assess_model_performance(model, X_test, y_test, prediction_date=None)`: Evaluates the model's performance using MSE and R2.
* `visualize_predictions(y_test, predictions, ticker, prediction_date=None)`: Plots the actual and predicted prices.
* `run_stock_prediction(ticker, start_date, end_date, prediction_date=None)`: Orchestrates the entire process.
* `if __name__ == "__main__":`: The entry point of the script, which gets user inputs and calls `run_stock_prediction()`.

## Important Considerations

* Data Source: The script relies on data from Yahoo Finance. Data availability and accuracy may vary.
* Model Limitations: The Random Forest Regressor is a powerful algorithm, but it's not guaranteed to provide accurate predictions of stock prices. The stock market is influenced by many complex and unpredictable factors.
* Feature Importance: You can explore the `feature_importances_` attribute of the trained Random Forest model to understand which features have the greatest influence on the predictions.
* Hyperparameter Tuning: The performance of the model can be improved by tuning the hyperparameters of the Random Forest Regressor (e.g., `n_estimators`, `max_depth`, `min_samples_split`). Consider using techniques like cross-validation to find optimal values.
* Time Series Analysis: For more advanced stock market forecasting, you might explore time series-specific models like ARIMA or LSTM neural networks.

## Author

* Rohit Karal

## Contributions

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.
