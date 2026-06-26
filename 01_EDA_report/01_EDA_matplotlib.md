##


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sns.set_theme(style="whitegrid", palette="muted")

notebook_dir = os.getcwd()
parent_dir = os.path.dirname(notebook_dir)
sys.path.append(parent_dir)
from config_path import cfg_path

# Read processed data from the previously written csv

df = pd.read_parquet(os.path.join(cfg_path.workspace_root, cfg_path.processed_data))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>^GSPC</th>
      <th>FEZ</th>
      <th>CL=F</th>
      <th>NG=F</th>
      <th>EURUSD=X</th>
      <th>USDCHF=X</th>
      <th>Brent_EIA</th>
      <th>^VIX</th>
      <th>^GSPC_log_ret</th>
      <th>FEZ_log_ret</th>
      <th>...</th>
      <th>^GSPC_drawdown</th>
      <th>FEZ_drawdown</th>
      <th>CL=F_drawdown</th>
      <th>NG=F_drawdown</th>
      <th>EURUSD=X_drawdown</th>
      <th>USDCHF=X_drawdown</th>
      <th>Brent_EIA_drawdown</th>
      <th>high_vol_regime</th>
      <th>bear_market_flag</th>
      <th>vol_spike_flag</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2004-01-05</th>
      <td>1122.219971</td>
      <td>17.768866</td>
      <td>33.779999</td>
      <td>6.827</td>
      <td>1.268698</td>
      <td>1.2320</td>
      <td>32.30</td>
      <td>17.49</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2004-01-06</th>
      <td>1123.670044</td>
      <td>17.862984</td>
      <td>33.700001</td>
      <td>7.082</td>
      <td>1.272103</td>
      <td>1.2319</td>
      <td>31.20</td>
      <td>16.73</td>
      <td>0.001291</td>
      <td>0.005283</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.002371</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000081</td>
      <td>-0.034649</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2004-01-07</th>
      <td>1126.329956</td>
      <td>17.615295</td>
      <td>33.619999</td>
      <td>6.878</td>
      <td>1.264095</td>
      <td>1.2395</td>
      <td>30.99</td>
      <td>15.50</td>
      <td>0.002364</td>
      <td>-0.013963</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.013963</td>
      <td>-0.004742</td>
      <td>-0.029228</td>
      <td>-0.006315</td>
      <td>0.000000</td>
      <td>-0.041169</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2004-01-08</th>
      <td>1131.920044</td>
      <td>17.967005</td>
      <td>33.980000</td>
      <td>7.094</td>
      <td>1.277498</td>
      <td>1.2266</td>
      <td>31.11</td>
      <td>15.61</td>
      <td>0.004951</td>
      <td>0.019769</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.010462</td>
      <td>-0.037463</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2004-01-09</th>
      <td>1121.859985</td>
      <td>17.808500</td>
      <td>34.310001</td>
      <td>7.287</td>
      <td>1.285892</td>
      <td>1.2212</td>
      <td>31.91</td>
      <td>16.75</td>
      <td>-0.008927</td>
      <td>-0.008861</td>
      <td>...</td>
      <td>-0.008927</td>
      <td>-0.008861</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.014828</td>
      <td>-0.013024</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python

print(df.shape)
df.describe()
```

    (5446, 61)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>^GSPC</th>
      <th>FEZ</th>
      <th>CL=F</th>
      <th>NG=F</th>
      <th>EURUSD=X</th>
      <th>USDCHF=X</th>
      <th>Brent_EIA</th>
      <th>^VIX</th>
      <th>^GSPC_log_ret</th>
      <th>FEZ_log_ret</th>
      <th>...</th>
      <th>^GSPC_drawdown</th>
      <th>FEZ_drawdown</th>
      <th>CL=F_drawdown</th>
      <th>NG=F_drawdown</th>
      <th>EURUSD=X_drawdown</th>
      <th>USDCHF=X_drawdown</th>
      <th>Brent_EIA_drawdown</th>
      <th>high_vol_regime</th>
      <th>bear_market_flag</th>
      <th>vol_spike_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5445.000000</td>
      <td>5445.000000</td>
      <td>...</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5446.000000</td>
      <td>5174.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2471.827719</td>
      <td>30.005463</td>
      <td>69.966306</td>
      <td>4.367990</td>
      <td>1.225246</td>
      <td>1.006229</td>
      <td>73.930911</td>
      <td>18.972591</td>
      <td>0.000333</td>
      <td>0.000237</td>
      <td>...</td>
      <td>-0.111945</td>
      <td>-0.373064</td>
      <td>-0.595285</td>
      <td>-0.832560</td>
      <td>-0.255113</td>
      <td>-0.284724</td>
      <td>-0.592003</td>
      <td>0.156445</td>
      <td>0.209512</td>
      <td>0.135099</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1487.875462</td>
      <td>9.754323</td>
      <td>21.341922</td>
      <td>2.272677</td>
      <td>0.127573</td>
      <td>0.128582</td>
      <td>24.385980</td>
      <td>8.527202</td>
      <td>0.011943</td>
      <td>0.016088</td>
      <td>...</td>
      <td>0.137373</td>
      <td>0.193812</td>
      <td>0.278593</td>
      <td>0.234732</td>
      <td>0.127654</td>
      <td>0.119783</td>
      <td>0.300204</td>
      <td>0.363310</td>
      <td>0.406997</td>
      <td>0.341862</td>
    </tr>
    <tr>
      <th>min</th>
      <td>676.530029</td>
      <td>12.939043</td>
      <td>-37.630001</td>
      <td>1.482000</td>
      <td>0.959619</td>
      <td>0.722800</td>
      <td>9.120000</td>
      <td>9.140000</td>
      <td>-0.127652</td>
      <td>-0.133124</td>
      <td>...</td>
      <td>-0.609717</td>
      <td>-0.693269</td>
      <td>-0.956126</td>
      <td>-0.994491</td>
      <td>-0.458849</td>
      <td>-0.479035</td>
      <td>-0.983682</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1288.160004</td>
      <td>23.375074</td>
      <td>53.330002</td>
      <td>2.764250</td>
      <td>1.117481</td>
      <td>0.916643</td>
      <td>55.982500</td>
      <td>13.480000</td>
      <td>-0.004092</td>
      <td>-0.006696</td>
      <td>...</td>
      <td>-0.180601</td>
      <td>-0.513760</td>
      <td>-0.811412</td>
      <td>-0.968406</td>
      <td>-0.363212</td>
      <td>-0.366042</td>
      <td>-0.859359</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1987.355042</td>
      <td>28.528316</td>
      <td>68.559998</td>
      <td>3.667000</td>
      <td>1.206891</td>
      <td>0.971800</td>
      <td>71.185000</td>
      <td>16.525001</td>
      <td>0.000737</td>
      <td>0.000834</td>
      <td>...</td>
      <td>-0.045937</td>
      <td>-0.446412</td>
      <td>-0.713647</td>
      <td>-0.943161</td>
      <td>-0.292859</td>
      <td>-0.327420</td>
      <td>-0.711058</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3263.622559</td>
      <td>33.214244</td>
      <td>85.547503</td>
      <td>5.551750</td>
      <td>1.321261</td>
      <td>1.061950</td>
      <td>90.437500</td>
      <td>21.700001</td>
      <td>0.005677</td>
      <td>0.007962</td>
      <td>...</td>
      <td>-0.009466</td>
      <td>-0.252489</td>
      <td>-0.538866</td>
      <td>-0.846780</td>
      <td>-0.160081</td>
      <td>-0.221715</td>
      <td>-0.421272</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6932.049805</td>
      <td>64.630753</td>
      <td>145.289993</td>
      <td>15.378000</td>
      <td>1.598798</td>
      <td>1.324800</td>
      <td>143.950000</td>
      <td>82.690002</td>
      <td>0.109572</td>
      <td>0.161562</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 61 columns</p>
</div>



## Price evolution over time 
I plot the historical price evolution of all assets in the basket (equity, energy, FX, VIX, Brent).

The goal is to visually identify the main market stress periods (2008 financial crisis, COVID-19 2020,

2022 energy crisis), which should later emerge as distinct regimes in the HMM and change point

detection models.


```python
price_cols = ["^GSPC", "FEZ", "CL=F", "NG=F", "EURUSD=X", "USDCHF=X", "^VIX", "Brent_EIA"]

fig, axes = plt.subplots(4, 2, figsize=(13, 12))
axes = axes.flatten()

for i, col in enumerate(price_cols):
    axes[i].plot(df.index, df[col], linewidth=0.9, color="#1D6FA3")
    axes[i].set_title(col, fontsize=11)
    axes[i].tick_params(axis='x', labelsize=8)

fig.suptitle("Prezzi storici per asset", fontsize=14)
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_4_0.png)
    


## 2. VIX index over time

The VIX measures the market's expected 30-day volatility, implied by S&P 500 options prices.
It is mean-reverting and spikes sharply during market stress rather than trending persistently
like price levels. Two reference thresholds are shown: 20 (moderate volatility) and 30 (acute stress),
commonly used as informal regime boundaries in practitioner research.


```python
fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(df.index, df["^VIX"], color="#5F5E5A", alpha=0.08)
ax.plot(df.index, df["^VIX"], color="#5F5E5A", linewidth=1.0, label="VIX")

ax.axhline(20, linestyle="--", color="#BA7517", linewidth=1, label="20 — vol moderata")
ax.axhline(30, linestyle="--", color="#A32D2D", linewidth=1, label="30 — stress")

ax.set_title("VIX Index — 2004–2025")
ax.set_xlabel("Date")
ax.set_ylabel("VIX level")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_6_0.png)
    


## 3. Log returns distribution

I examine the distributional properties of daily log returns, focusing on the S&P 500 as
the primary benchmark. The computed Jarque-Bera test
indicated departure from normality; here I visualize that departure directly through
a histogram with an overlaid normal density and a QQ-plot.


```python
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

col = "^GSPC_log_ret"
returns = df[col].dropna()

jb_stat, jb_pval = stats.jarque_bera(returns)
print(f"Jarque-Bera test — {col}")
print(f"Statistic: {jb_stat:.2f}, p-value: {jb_pval:.4g}")
print("Normal" if jb_pval > 0.05 else "Reject normality (non-normal distribution)")

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(returns, bins=100, density=True, color="#1D6FA3", alpha=0.7, label="Log returns")

x_range = np.linspace(returns.min(), returns.max(), 200)
normal_curve = stats.norm.pdf(x_range, returns.mean(), returns.std())
ax.plot(x_range, normal_curve, color="red", linewidth=1.5, label="Normal distribution")

ax.set_title(f"Distribution of log returns — {col}")
ax.legend()
plt.tight_layout()
plt.show()

# QQ-plot for log returns
sm.qqplot(returns, line="s")
plt.title(f"QQ-plot — {col}")
plt.show()
```

    Jarque-Bera test — ^GSPC_log_ret
    Statistic: 41292.75, p-value: 0
    Reject normality (non-normal distribution)



    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_8_1.png)
    



    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_8_2.png)
    


## 5. Correlation matrix

I compute the pairwise correlation matrix of daily log returns across all assets.
This provides a static view of co-movement and helps sanity-check the dataset against
known relationships (e.g. positive correlation between S&P 500 and Euro Stoxx 50).


```python
log_ret_cols = [c for c in df.columns if c.endswith("_log_ret")]
corr_matrix = df[log_ret_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation matrix — log returns")
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_10_0.png)
    


## 6. Rolling volatility vs VIX

I plot the 21-day realized volatility of the S&P 500 alongside the VIX index.
Since the VIX is a forward-looking, options-implied measure while realized volatility is
backward-looking, I expect them to move together but not coincide exactly — divergences
between the two are themselves informative about market regime shifts.


```python
fig, ax1 = plt.subplots(figsize=(11, 5))

ax1.plot(df.index, df["^GSPC_vol_21d"], color="#1D9E75", label="SPX realized vol (21d)")
ax1.set_ylabel("SPX realized vol (21d)", color="#1D9E75")
ax1.tick_params(axis='y', labelcolor="#1D9E75")
ax1.set_xlabel("Date")

ax2 = ax1.twinx()
ax2.plot(df.index, df["^VIX"], color="#5F5E5A", label="VIX")
ax2.set_ylabel("VIX", color="#5F5E5A")
ax2.tick_params(axis='y', labelcolor="#5F5E5A")

fig.suptitle("Realized volatility (SPX, 21d) vs VIX")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88))
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_12_0.png)
    


## 7. Drawdown

I compute the drawdown from peak for the S&P 500, i.e. the percentage decline from the
highest cumulative value reached so far. This highlights both the depth and duration of
major stress periods, complementing the volatility-based view above.


```python
fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(df.index, df["^GSPC_drawdown"], color="#A32D2D", alpha=0.4)
ax.plot(df.index, df["^GSPC_drawdown"], color="#A32D2D", linewidth=0.8)
ax.set_title("SPX drawdown from peak")
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_14_0.png)
    


## 8. Rolling correlation: S&P 500 vs VIX

I examine the 63-day rolling correlation between S&P 500 and VIX log returns. This
correlation is structurally negative under normal market conditions, but tends to
compress toward zero during periods of acute stress — making it a useful regime
indicator in its own right.


```python
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df.index, df["spx_vix_rolling_corr"], color="#1D6FA3", linewidth=1.0)
ax.axhline(0, linestyle="--", color="gray", linewidth=1)
ax.set_title("Rolling correlation — SPX vs VIX (63d)")
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_16_0.png)
    


## 9. Sub-period statistics

I split the sample into five sub-periods (pre-GFC, the 2008-2009 financial crisis,
2010-2019, the 2020-2022 COVID/energy crisis window, and 2023-2025) and compute summary
statistics of S&P 500 log returns for each. This gives an empirical, model-free sense of
how many distinct regimes the data might contain, ahead of formal regime detection.


```python
periods = {
    "2004-2007": ("2004-01-01", "2007-12-31"),
    "2008-2009 (GFC)": ("2008-01-01", "2009-12-31"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-2022 (COVID+energy)": ("2020-01-01", "2022-12-31"),
    "2023-2025": ("2023-01-01", "2025-12-31"),
}

records = []
for label, (start, end) in periods.items():
    subset = df.loc[start:end, "^GSPC_log_ret"].dropna()
    records.append({
        "period": label,
        "mean_daily_ret": subset.mean(),
        "ann_vol": subset.std() * np.sqrt(252),
        "skew": subset.skew(),
        "kurtosis": subset.kurt()
    })

print(pd.DataFrame(records).set_index("period"))
```

                              mean_daily_ret   ann_vol      skew   kurtosis
    period                                                                 
    2004-2007                       0.000272  0.121033 -0.317141   1.807681
    2008-2009 (GFC)                -0.000566  0.353882 -0.114471   4.203413
    2010-2019                       0.000427  0.148009 -0.551107   4.493974
    2020-2022 (COVID+energy)        0.000233  0.257371 -0.755528  10.849529
    2023-2025                       0.000794  0.150586  0.332480  14.339260


## 10. Stationarity check (Augmented Dickey-Fuller test)

Most time series models (HMM, GARCH) assume stationary input. I verify empirically what
is normally assumed: price levels should be non-stationary (they follow a random-walk-like
process with trend), while log returns should be stationary. The ADF test formalizes this —
a low p-value rejects the null hypothesis of a unit root, i.e. confirms stationarity.


```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna())
    verdict = "stationary" if result[1] < 0.05 else "non-stationary"
    print(f"{name:25s} ADF stat={result[0]:8.3f}  p-value={result[1]:.4f}  -> {verdict}")

adf_test(df["^GSPC"], "SPX price level")
adf_test(df["^GSPC_log_ret"], "SPX log returns")
adf_test(df["^VIX"], "VIX level")
adf_test(df["Brent_EIA"], "Brent price level")
```

    SPX price level           ADF stat=   2.548  p-value=0.9991  -> non-stationary
    SPX log returns           ADF stat= -15.950  p-value=0.0000  -> stationary
    VIX level                 ADF stat=  -5.796  p-value=0.0000  -> stationary
    Brent price level         ADF stat=  -2.948  p-value=0.0400  -> stationary


## 11. Autocorrelation: returns vs squared returns

Market efficiency implies that raw returns should show little to no autocorrelation —
past returns shouldn't predict future ones. However, squared returns (a proxy for variance)
are expected to show strong, persistent autocorrelation. This is the empirical signature of
volatility clustering: large moves tend to be followed by large moves, regardless of sign.


```python
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df["^GSPC_log_ret"].dropna(), lags=40, ax=axes[0])
axes[0].set_title("ACF — SPX log returns")
plot_acf(df["^GSPC_log_ret"].dropna()**2, lags=40, ax=axes[1])
axes[1].set_title("ACF — SPX squared log returns")
plt.tight_layout()
plt.show()
```


    
![png](01_EDA_matplotlib_files/01_EDA_matplotlib_22_0.png)
    


## 12. Ljung-Box test on squared returns

I formally test what the ACF plot above suggests visually: whether squared returns exhibit
statistically significant autocorrelation at multiple lags. A low p-value confirms the
presence of volatility clustering beyond what could be attributed to random noise.


```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_result = acorr_ljungbox(df["^GSPC_log_ret"].dropna()**2, lags=[5, 10, 20], return_df=True)
print(lb_result)
```

            lb_stat  lb_pvalue
    5   3138.879570        0.0
    10  5343.273677        0.0
    20  7536.465902        0.0


## 13. ARCH test (conditional heteroskedasticity)

Engle's ARCH test directly examines whether the variance of returns is time-varying and
predictable from its own past — i.e. whether the series exhibits autoregressive conditional
heteroskedasticity. A significant result is the formal statistical justification for moving
to a GARCH-family model rather than assuming constant volatility.


```python
from statsmodels.stats.diagnostic import het_arch

arch_stat, arch_pval, f_stat, f_pval = het_arch(df["^GSPC_log_ret"].dropna())
print(f"ARCH test (LM): stat={arch_stat:.2f}, p-value={arch_pval:.4f}")
print(f"ARCH test (F):  stat={f_stat:.2f}, p-value={f_pval:.4f}")
```

    ARCH test (LM): stat=1543.86, p-value=0.0000
    ARCH test (F):  stat=215.20, p-value=0.0000



