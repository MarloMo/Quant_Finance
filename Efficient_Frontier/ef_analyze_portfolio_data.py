import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import rf as rf
import pandas_datareader as pdr

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
TICKERS_PATH = "../Tickers/my_sec_tickers.txt"
START_DATE = "2018-01-01"
END_DATE = "2025-05-07"
RISK_FREE_RATE_ANNUAL = 0.0433  # Approximate 3-month T-bill rate
RISK_FREE_RATE_QUARTERLY = RISK_FREE_RATE_ANNUAL / 4  # Match quarterly returns
# API_KEY = 
# ----------------------------------------
# LOAD TICKERS
# ----------------------------------------
try:
    tickers = np.loadtxt(TICKERS_PATH, dtype=str).tolist()
except Exception as e:
    raise ValueError(f"Error loading tickers from {TICKERS_PATH}: {e}")

# ----------------------------------------
# DOWNLOAD DATA
# ----------------------------------------
# data = yf.download(
#     tickers, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=False
# )["Adj Close"]

# Download adjusted close prices from Tiingo (use Yahoo-yfiance as alternative)
tiingo_data = pdr.get_data_tiingo(
    tickers, start=START_DATE, end=END_DATE, api_key=API_KEY
)
data = tiingo_data.reset_index().pivot(
    index="date", columns="symbol", values="adjClose"
)

if data.isnull().values.any():
    data.ffill(inplace=True)
    data.bfill(inplace=True)

# ----------------------------------------
# RESAMPLE TO QUARTERLY DATA
# ----------------------------------------
quarterly_data = data.resample("QE").last()
quarterly_data.columns = tickers  # Ensure columns match tickers

# ----------------------------------------
# CALCULATE RETURNS AND COVARIANCE
# ----------------------------------------
quarterly_returns = quarterly_data.pct_change().dropna()
expected_returns_avg = quarterly_returns.mean()
returns_cov = quarterly_returns.cov()

print("\nQuarterly Returns Columns:")
print(quarterly_returns.columns.tolist(), "\n")

print("Expected Average Quarterly Returns:")
print(expected_returns_avg, "\n")

# ----------------------------------------
# OUTLIER CHECK (OPTIONAL)
# ----------------------------------------
outliers = quarterly_returns[(quarterly_returns > 1) | (quarterly_returns < -1)]
if not outliers.dropna(how="all").empty:
    print("Warning: Outliers detected in returns data.")

# ----------------------------------------
# RANDOM PORTFOLIO GENERATION
# ----------------------------------------
random_portfolios = rf.return_portfolios_original(expected_returns_avg, returns_cov)
print("Random portfolios generated (for exploratory analysis).\n")

# ----------------------------------------
# OPTIMIZED PORTFOLIO AND EFFICIENT FRONTIER
# ----------------------------------------
optimal_weights, returns, risks, frontier_weights = rf.optimal_portfolio(
    quarterly_returns, check_labels=True
)

# Sort risks and returns for smooth plotting
sorted_indices = np.argsort(risks)
risks = np.array(risks)[sorted_indices]
returns = np.array(returns)[sorted_indices]

# Efficient frontier composition insight
print("=" * 100)
print("Efficient Frontier Weights:\n")
x = 50  # Choose a portfolio index along the frontier
print(f"Weights at start (index 0):\n{frontier_weights[0]}\n")
print(f"Weights at index {x}:\n{frontier_weights[x]}\n")
print(f"Weights at end (index -1):\n{frontier_weights[-1]}\n")
print(f"Optimal Portfolio Weights:\n{optimal_weights.flatten()}\n")
print("=" * 100)

# ----------------------------------------
# SHARPE RATIO OPTIMIZATION
"""
Now you want to maximize returns relative to standard deviation. 
This essentially means optimizing for the highest Sharpe ratio, which is the 
best risk-adjusted return.

Sharpe Ratio Interpretation:
SR < 1.0       : Suboptimal; risk is high relative to return
1.0 < SR < 1.5 : Good; reasonable balance between risk and return
1.5 < SR < 2.0 : Very good, strong risk-adjusted returns
SR > 2.0       : Excellent, higiht return for the amount of risk taken
"""
# ----------------------------------------
sharpe_ratios = [(r - RISK_FREE_RATE_QUARTERLY) / v for r, v in zip(returns, risks)]
max_sharpe_idx = np.argmax(sharpe_ratios)

print(f"Max Sharpe Ratio Portfolio:")
print(f"Index: {max_sharpe_idx}")
print(f"Expected Return: {returns[max_sharpe_idx]:.3f}")
print(f"Volatility: {risks[max_sharpe_idx]:.3f}")
print(f"Weights:\n{frontier_weights[max_sharpe_idx]}")
print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.3f}\n")

# ----------------------------------------
# PLOT EFFICIENT FRONTIER
# ----------------------------------------
plt.figure(figsize=(10, 8))
plt.plot(risks, returns, "y-o", label="Efficient Frontier", markersize=5)

# Highlight special portfolios
plt.scatter(
    risks[0], returns[0], c="red", label="Min-Risk Portfolio", s=120, marker="X"
)
plt.scatter(
    risks[-1], returns[-1], c="green", label="Max-Return Portfolio", s=120, marker="X"
)
plt.scatter(
    risks[max_sharpe_idx],
    returns[max_sharpe_idx],
    c="black",
    label="Max Sharpe Portfolio",
    s=120,
    marker="X",
)

# Annotate key points
plt.axvline(x=risks[0], color="red", linestyle="--")
plt.axhline(y=returns[0], color="red", linestyle="--")
plt.axvline(x=risks[-1], color="green", linestyle="--")
plt.axhline(y=returns[-1], color="green", linestyle="--")
plt.axvline(x=risks[max_sharpe_idx], color="black", linestyle="--")
plt.axhline(y=returns[max_sharpe_idx], color="black", linestyle="--")

plt.title("Efficient Frontier with Sharpe Optimization", fontsize=20)
plt.legend()
plt.xlabel("Volatility (Standard Deviation)", fontsize=16)
plt.ylabel("Expected Returns", fontsize=16)
plt.xticks(fontsize=16)  # Increase x-axis numbers
plt.yticks(fontsize=16)  # Increase y-axis numbers
plt.grid(True)
# plt.tight_layout()
plt.savefig("efficient_frontier_plot.pdf")


"""
NEXT STEPS - IGNORE FOR NOW:
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
