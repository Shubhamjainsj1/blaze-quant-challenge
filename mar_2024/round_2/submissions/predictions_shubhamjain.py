# %%
pip install instructions

# %%
"""
For round 2, you have to predict ETH price in USD between Mar 16th and Feb 22nd 2024 (inclusive).
Each team will have to make 7 predictions; one for each day. The prediction has to be
within a range. Each range is inclusive of the lower bound and exclusive of the higher bound.
E.g. pr_2000_2025 is [2000, 2025).

Note: the price ranges might be updated closer to Mar 16.
Please copy the example to add your predictions.
We will call the predictions function on Feb Mar 16 12:00 AM PST. After that, the predictions
will not be altered.

We will take the closing price on the day as the correct price. The source for the closing
price will be https://coinmarketcap.com/currencies/ethereum/. Coin market cap provides
prices at 5 min intervals. We will take the price at 11:55 pm PST as the closing price.
"""
from enum import Enum


class ETHPriceRanges(Enum):
    pr_2000_2025 = 1
    pr_2025_2050 = 2
    pr_2050_2075 = 3
    pr_2075_2100 = 4
    pr_2100_2125 = 5
    pr_2125_2150 = 6
    pr_2150_2175 = 7
    pr_2300_2325 = 8
  


class ARBPriceRanges(Enum):
    pr_135_140 = 1
    pr_140_145 = 2
    pr_145_150 = 3
    pr_150_155 = 4
    pr_155_160 = 5
    pr_160_165 = 6
    pr_165_170 = 7
    pr_180_185 = 8


class LINKPriceRanges(Enum):
    pr_1800_1825 = 1
    pr_1825_1850 = 2
    pr_1850_1875 = 3
    pr_1875_1900 = 4
    pr_1900_1925 = 5
    pr_1925_1950 = 6
    pr_1950_1975 = 7
    

# %%
import numpy as np
from collections.abc import Iterable
def simple_moving_average(data, window_size=7):
    """
    Computes the simple moving average of the given data using the specified window size.
    """
    if len(data) < window_size:
        raise ValueError("Window size is larger than the length of the data.")
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')
def exponential_moving_average(data, asset, window_size=7, alpha=0.2):
    """
    Computes the exponential moving average of the given data using the specified window size and smoothing factor.
    """
    ema = [data[0] - 200 if asset == 'ETH' else data[0]]  # Initialize the EMA with the value of the first day
    for i in range(1, len(data)):
        if i < 3:  # Subtract 200 for the first 3 days
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1] - 200 if asset == 'ETH' else alpha * data[i] + (1 - alpha) * ema[i-1])
        elif i>2 and i<5:  # Subtract 100 for the next 4 days
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1] - 50 if asset == 'ETH' else alpha * data[i] + (1 - alpha) * ema[i-1])
        else:
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1] - 30 if asset == 'ETH' else alpha * data[i] + (1 - alpha) * ema[i-1])
    return ema

def predict_prices(data, asset, window_size=7, alpha=0.2):
    """
    Predicts future prices for the next 'num_days' based on historical data 
    using a simple moving average model.
    """
    predictions = []
    for i in range(len(data) - window_size + 1):
        window_data = data[i:i+window_size]
        sma = simple_moving_average(window_data, window_size=window_size)
        ema = exponential_moving_average(window_data, asset, window_size=window_size, alpha=alpha)
        
    return sma,ema

def predictions_ETH():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from ETHPriceRanges
    """
    return [
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
        ETHPriceRanges.pr_2300_2325,
    ]

def predictions_ARB():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from ARBPriceRanges
    """
    return [
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
        ARBPriceRanges.pr_180_185,
    ]


def predictions_LINK():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from LINKPriceRanges
    """
    return [
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
        LINKPriceRanges.pr_1875_1900,
    ]

"""
DO NOT REMOVE
"""
preds_ETH = predictions_ETH()
assert len(preds_ETH) == 7
assert all([isinstance(val, ETHPriceRanges) for val in preds_ETH])

preds_ARB = predictions_ARB()
assert len(preds_ARB) == 7
assert all([isinstance(val, ARBPriceRanges) for val in preds_ARB])

preds_LINK = predictions_LINK()
assert len(preds_LINK) == 7
assert all([isinstance(val, LINKPriceRanges) for val in preds_LINK])


# %%
import pandas as pd
import csv
import matplotlib.pyplot as plt

# %%
# Generate predictions for each asset
preds_ETH = predictions_ETH()
preds_ARB = predictions_ARB()
preds_LINK = predictions_LINK()

# Example historical data (replace with your actual data)
data_eth = pd.read_csv(r"C:\Users\Shubham\Desktop\blaze\ETH_1M_graph_coinmarketcap - ETH_1M_graph_coinmarketcap.csv.csv")
data_eth = data_eth[['close']]
data_arb = pd.read_csv(r"C:\Users\Shubham\Desktop\blaze\ARB_1M_graph_coinmarketcap - ARB_1M_graph_coinmarketcap.csv.csv")
data_arb = data_arb[['close']]
data_link = pd.read_csv(r"C:\Users\Shubham\Desktop\blaze\LINK_ETH Binance Historical Data - LINK_ETH Binance Historical Data.csv.csv")
data_link = data_link[['close']]
# data_link = pd.read_csv(r"C:\Users\Shubham\Desktop\blaze\LINK_ETH Binance Historical Data.csv")
# data_link = data_link[['close']]
# data_arb = pd.read_csv()
# historical_data = [2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600]

# Convert DataFrame columns to lists of numerical values
data_eth_values = data_eth['close'].tolist()
data_arb_values = data_arb['close'].tolist()
data_link_values = data_link['close'].tolist()
# Calculate moving averages for each asset
sma_eth, ema_eth = predict_prices(data_eth_values, 'ETH', window_size=7, alpha=0.2)
sma_arb, ema_arb = predict_prices(data_arb_values, 'ARB', window_size=7, alpha=0.4)
sma_link, ema_link = predict_prices(data_link_values, 'LINK', window_size=7, alpha=0.3)

print("EMA for ETH:", ema_eth)

print("EMA for ARB:", ema_arb)

print("EMA for LINK:", ema_link)




