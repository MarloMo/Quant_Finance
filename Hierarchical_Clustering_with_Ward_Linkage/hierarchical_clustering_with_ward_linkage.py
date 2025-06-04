import numpy as np
import yfinance as yf
import riskfolio as rp
import seaborn as sns
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# api_key = 

# Load tickers from text file
my_tickers = "../Tickers/my_sec_tickers.txt"
tickers_file = np.loadtxt(my_tickers, dtype=str)
tickers = tickers_file.tolist()

# Download adjusted close prices from Tiingo (use Yahoo-yfiance as alternative)
raw_data = pdr.get_data_tiingo(
    tickers, start="2018-01-01", end="2025-04-16", api_key=api_key
)
adj_close_data = raw_data.reset_index().pivot(
    index="date", columns="symbol", values="adjClose"
)

# Optional: Check for missing values
print("Missing values per ticker:")
print(adj_close_data.isnull().sum())
print()

# Resample to monthly data (more date points vs quarterly)
monthly_data = adj_close_data.resample("ME").last()

# Compute monthly returns
monthly_returns = monthly_data.pct_change().dropna()

# Optional sanity check
print("Monthly returns statistics:")
print(monthly_returns.describe())
print()

# Calculate correlation matrix
correlation_matrix = monthly_returns.corr()
print("Correlation matrix:")
print(correlation_matrix)

# Plot Riskfolio clusters
num_clusters = min(monthly_returns.shape[1], 5)

rp.plot_clusters(
    returns=monthly_returns,
    codependence="pearson",
    linkage="ward",
    k=num_clusters,
    max_k=10,
    leaf_order=True,
    dendrogram=True,
    ax=None,
)

plt.savefig("clusters_plot.pdf", format="pdf", bbox_inches="tight")
print("Saved: clusters_plot.pdf")

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)

plt.title("Correlation Matrix Heatmap")
plt.savefig("correlation_heatmap.pdf", format="pdf", bbox_inches="tight")
print("Saved: correlation_heatmap.pdf")
