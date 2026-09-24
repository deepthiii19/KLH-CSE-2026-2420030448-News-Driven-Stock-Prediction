import yfinance as yf
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def download_stock_data(
    ticker="AAPL",
    start="2020-01-01",
    end="2025-01-01"
):
    """
    Download historical stock market data.
    """

    print("=" * 60)
    print("DOWNLOADING STOCK DATA")
    print("=" * 60)

    print(f"Ticker: {ticker}")
    print(f"Start: {start}")
    print(f"End: {end}")

    stock = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False
    )

    if stock.empty:
        raise ValueError(
            "No stock data was downloaded."
        )

    return stock


if __name__ == "__main__":

    stock_data = download_stock_data()

    print("\nDownloaded data shape:")
    print(stock_data.shape)

    print("\nFirst 5 rows:")
    print(stock_data.head())

    print("\nLast 5 rows:")
    print(stock_data.tail())

    print("\nColumns:")
    print(stock_data.columns)