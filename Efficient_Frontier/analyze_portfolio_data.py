import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rf as rf

"""
IMPORT / DOWNLOAD THE DATA
"""

tickers = ["VOO", "XLK", "QQQ", "IJH", "IWM"]

# Download the historical adjusted close prices for tickers
data = yf.download(tickers, start="2015-01-01", end="2025-01-08", interval="1d")[
    "Adj Close"
]

# # Check for missing values and handle them
# print("Missing values before filling:")
# print(data.isnull().sum())  # Display missing values per ticker

# # Handle missing values using forward fill and backward fill
# data.ffill(inplace=True)  # Forward fill missing values
# data.bfill(inplace=True)  # Backward fill remaining missing values

# print("Missing values after filling:")
# print(data.isnull().sum())  # Verify no missing values remain


"""
RESAMPLE TO QUARTERLY DATA
"""
# Resample data to quarterly frequency ('QE' for end of quarter)
quarterly_data = data.resample("QE").last()

# Rename columns to reflect tickers
quarterly_data.columns = tickers

# # Validate the resampling
# print("Quarterly data preview:")
# print(quarterly_data.head())

"""
CALCULATE FINANCIAL STATISTICS
"""

# Calculate quarterly returns (percentage change)
quarterly_returns = quarterly_data.pct_change()
print("")
print("quartlery_returns Columns:")
print(quarterly_returns.columns, "\n")

# Handle missing returns (e.g. first row will NaN)
quarterly_returns.dropna(inplace=True)

# Caluclate average expected return for each ticker
expected_returns_avg = quarterly_returns.mean()

# Outlier analysis (optional)
# print("Quarterly Returns Summary Statistics:")
# print(quarterly_returns.describe())

outliers = quarterly_returns[(quarterly_returns > 1) | (quarterly_returns < -1)]
# print("Outliers (Returns > 1 or < -1):")
# print(outliers)
# if outliers.isna().all().all():
#     print("No outliers detected in the dataset.")
# else:
#     print("Outliers detected:")
#     print(outliers.dropna(how="all"))  # Show rows with at least one non-NaN value

# Display expected average returns
print("Expected Average Quarterly Returns:")
print(expected_returns_avg, "\n")

# Covariance Matrix
returns_cov = quarterly_returns.cov()
print("COVARIANCE MATRIX:")
print(returns_cov.head(20), "\n")

""" 
OPTIMIZED PORTFOLIO
"""
# Returns an array of N portfolios with different Returns, Volatility, and Weights in
# array
random_portfolios = rf.return_portfolios_original(expected_returns_avg, returns_cov)
# print(random_portfolios.head())
print("The goal is to minimize volatility and maximize expected returns. \n")

# Calculate the efficient frontier
optimal_weights, returns, risks, weights = rf.optimal_portfolio(
    quarterly_returns[1:], check_labels=True
)

random_portfolios.plot.scatter(x="Volatility", y="Returns", fontsize=12)
print("")
print(f"Optimal Portfolio Weights:\n {weights[99]} \n")

# Sort the data points (the function return is reveresed) - FIX LATER
# AND figure out why the function is doing this.
sorted_indices = np.argsort(risks)
risks = np.array(risks)[sorted_indices]
returns = np.array(returns)[sorted_indices]

# Analysis of portfolio composition
# This will give insights into how asset allocation changes across the frontier
print("=" * 100)
print("Efficient Frontier Weights: \n")
x = 50  # 0 < x < 100
print(f"EF_weight[0] =\n {weights[0]} \n")
print(f"EF_weight[{x}] =\n {weights[x]} \n")
print(f"EF_weight[-1] =\n {weights[-1]}")
print("=" * 100)

# Plot the efficient frontier
plt.plot(risks, returns, "y-o", label="Efficient Frontier")
# Highlight min-risk and max-return portfolios
plt.scatter(
    risks[0], returns[0], c="red", label="Min-Risk Portfolio", s=180, marker="X"
)
plt.scatter(
    risks[-1], returns[-1], c="green", label="Max-Return Portfolio", s=180, marker="X"
)
# Add vertical and horizontal lines for min-risk portfolio
plt.axvline(x=risks[0], color="red", linestyle="--")
plt.axhline(y=returns[0], color="red", linestyle="--")
# Add vertical and horizontal lines for max-return portfolio
plt.axvline(x=risks[-1], color="green", linestyle="--")
plt.axhline(y=returns[-1], color="green", linestyle="--")
plt.legend()
plt.ylabel("Expected Returns", fontsize=14)
plt.xlabel("Volatility (Std. Deviation)", fontsize=14)
plt.title("Efficient Frontier", fontsize=24)
plt.savefig("efficient_frontier_plot.pdf")


"""
NEXT STEPS:
3. Overlay Sharpe Ratio Analysis 
	* Add the Capital Market Line (CML) to show the Sharpe ratio for each portfolio 
      if you have a risk-free rate.
    * Learn the meaning of this.
    
4. Start to diversify so that I can increase the returns
    * CONSIDER:
    * volatility (x-axis) ranges from approximately 0.08 to 0.10, indicating relatively 
      low levels of portfolio risk.
	•	The expected returns (y-axis) range from approximately 0.035 to 0.05, which 
        suggests the portfolios are moderately aggressive but not extremely 
        high-return or high-risk.
    * Find out if the weights are in percent and how to represent exponential percents?
    
"""
