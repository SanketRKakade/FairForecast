import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
        if not stock_data.empty:
            stock_data.to_csv(f'stock_data/{ticker}.csv')
            print(f'Successfully downloaded data for {ticker} and saved to stock_data/{ticker}.csv')
        else:
            print(f'No data found for {ticker}. Please check the ticker symbol.')
    except Exception as e:
        print(f'Error downloading data for {ticker}: {e}')

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(stock_data):
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Close'])

    stock_data['Direction'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)

    stock_data = stock_data.dropna()

    X = stock_data[['MA50', 'RSI', 'Close', 'Volume']]
    y = stock_data['Direction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    ticker = 'AAPL'  # Change this to any stock ticker you want to analyze
    download_stock_data(ticker)

    # Load the data from CSV after downloading
    try:
        stock_data = pd.read_csv(f'stock_data/{ticker}.csv', index_col='Date', parse_dates=True)
        X_train, X_test, y_train, y_test = preprocess_data(stock_data)
        print('Preprocessing completed successfully.')
    except FileNotFoundError:
        print(f'File stock_data/{ticker}.csv not found. Ensure that data has been downloaded successfully.')
