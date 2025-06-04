import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def compare_to_index(
    security_ticker, index_ticker, start_date="2023-01-01", end_date="2024-01-01"
):
    # Download historical data
    data = yf.download(
        [security_ticker, index_ticker],
        start=start_date,
        end=end_date,
        auto_adjust=False,
    )["Adj Close"]

    # Normalize both to start at 100
    normalized = data / data.iloc[0] * 100

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(normalized[security_ticker], label=f"{security_ticker}", linewidth=2)
    plt.plot(
        normalized[index_ticker], label=f"{index_ticker}", linewidth=2, linestyle="--"
    )
    plt.title(f"{security_ticker} vs {index_ticker} Performance")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Start = 100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_to_index.pdf")


# Example usage:
# Compare QQQ to NASDAQ-100 Index (^NDX)
compare_to_index("QQQ", "^NDX")

# You can also try:
# compare_to_index("AAPL", "^GSPC")  # Apple vs S&P 500
# compare_to_index("TSLA", "^DJI")   # Tesla vs Dow Jones
