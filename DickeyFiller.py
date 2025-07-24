import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Generate a stationary time series (White Noise)
np.random.seed(42)
n = 100
white_noise = np.random.normal(0, 1, n)

# Perform Augmented Dickey-Fuller (ADF) test

adf_result_st = adfuller(white_noise)

# Add linear trend to make it non-stationary

trend = 0.2 * np.arange(n)
nonstationary = trend + white_noise

adf_result_nst = adfuller(nonstationary)

checkSt = [adf_result_st,adf_result_nst]

for i in checkSt:
    print(f"ADF Statistic: {i[0]}")
    print(f"P-Value: {i[1]}")
    if i[1] < 0.05:
        print("Stationary process (Reject Null Hypothesis)")
    else:
        print("Non-Stationary process (Fail to Reject Null Hypothesis)")


