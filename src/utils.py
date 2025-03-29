import yfinance as yf

def get_live_data(ticker):
    try:
        data = yf.Ticker(ticker)
        return data  # Return the entire data object instead of just the price
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None