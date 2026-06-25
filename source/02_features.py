import yaml
from box import Box
import os
import pandas as pd

import numpy as np
from scipy import stats
# import pyarrow
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_path import cfg_path

# ── Column groups ────────────────────────────────────────────────────────────
PRICE_COLS   = ["^GSPC", "FEZ", "CL=F", "NG=F", "EURUSD=X", "USDCHF=X", "Brent_EIA"]
RETURN_COLS  = PRICE_COLS          # log-return columns share the same names after transform
SPX_COL      = "^GSPC"
VIX_COL      = "^VIX"

WINDOW_SHORT = 21   # ~1 month
WINDOW_LONG  = 63   # ~3 months
TRADING_DAYS = 252


class FeatureEngineer:
    """
    Builds the enriched feature set (silver → gold layer) starting from raw
    daily prices produced by 01_ingest.py.
-
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw price dataframe with a DatetimeIndex and columns matching PRICE_COLS + [VIX_COL].
    vix_high_threshold : float
        VIX level above which the market is flagged as high-volatility. Default 25.
    bear_ma_window : int
        Look-back window (days) for the SPX moving average used in the bear-market flag. Default 200.
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,
        vix_high_threshold: float = 25.0,
        bear_ma_window: int = 200,
    ):
        self.df_raw             = df_raw.copy()
        self.vix_high_threshold = vix_high_threshold
        self.bear_ma_window     = bear_ma_window

        # Ensure DatetimeIndex
        if not isinstance(self.df_raw.index, pd.DatetimeIndex):
            self.df_raw.index = pd.to_datetime(self.df_raw.index)

        self.df_features: pd.DataFrame = pd.DataFrame(index=self.df_raw.index)

    # ── 1. Log returns ───────────────────────────────────────────────────────

    def compute_log_returns(self) -> pd.DataFrame:
        """Compute daily log returns for all price columns."""
        prices = self.df_raw[PRICE_COLS]
        log_ret = np.log(prices / prices.shift(1))
        log_ret.columns = [f"{c}_log_ret" for c in PRICE_COLS]
        return log_ret

    # ── 2. Rolling volatility ────────────────────────────────────────────────

    def compute_rolling_volatility(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling standard deviation of log returns (short and long window).
        Also computes annualized realized volatility (short window × sqrt(252)).
        """
        frames = {}
        for col in log_returns.columns:
            asset = col.replace("_log_ret", "")
            frames[f"{asset}_vol_{WINDOW_SHORT}d"]    = log_returns[col].rolling(WINDOW_SHORT).std()
            frames[f"{asset}_vol_{WINDOW_LONG}d"]     = log_returns[col].rolling(WINDOW_LONG).std()
            frames[f"{asset}_realized_vol"]           = (
                log_returns[col].rolling(WINDOW_SHORT).std() * np.sqrt(TRADING_DAYS)
            )
        return pd.DataFrame(frames, index=log_returns.index)

    # ── 3. Rolling momentum ──────────────────────────────────────────────────

    def compute_rolling_momentum(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean of log returns as a proxy for momentum."""
        frames = {}
        for col in log_returns.columns:
            asset = col.replace("_log_ret", "")
            frames[f"{asset}_mom_{WINDOW_SHORT}d"] = log_returns[col].rolling(WINDOW_SHORT).mean()
            frames[f"{asset}_mom_{WINDOW_LONG}d"]  = log_returns[col].rolling(WINDOW_LONG).mean()
        return pd.DataFrame(frames, index=log_returns.index)

    # ── 4. Rolling SPX/VIX correlation ──────────────────────────────────────

    def compute_spx_vix_correlation(self, log_returns: pd.DataFrame) -> pd.Series:
        """
        Rolling correlation between S&P 500 and VIX log returns.
        Typically negative; spikes toward zero / positive signal stress regimes.
        """
        spx_ret = log_returns[f"{SPX_COL}_log_ret"]
        vix_ret = np.log(self.df_raw[VIX_COL] / self.df_raw[VIX_COL].shift(1))
        rolling_corr = spx_ret.rolling(WINDOW_LONG).corr(vix_ret)
        return rolling_corr.rename("spx_vix_rolling_corr")

    # ── 5. Drawdown ──────────────────────────────────────────────────────────

    def compute_drawdown(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Drawdown from peak for each asset, computed from cumulative returns.
        Value is 0 at all-time highs, negative otherwise (e.g. -0.30 = 30% drawdown).
        """
        frames = {}
        for col in log_returns.columns:
            asset      = col.replace("_log_ret", "")
            cum_ret    = (1 + log_returns[col].fillna(0)).cumprod()
            rolling_max = cum_ret.cummax()
            frames[f"{asset}_drawdown"] = cum_ret / rolling_max - 1
        return pd.DataFrame(frames, index=log_returns.index)

    # ── 6. Baseline regime labels ────────────────────────────────────────────

    def compute_baseline_regime_labels(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Simple rule-based regime flags.
        These serve as interpretable baselines before HMM / change-point models.

        - high_vol_regime   : 1 when VIX > vix_high_threshold
        - bear_market_flag  : 1 when SPX price < its {bear_ma_window}-day moving average
        - vol_spike_flag    : 1 when realized vol of SPX is in the top decile (rolling 252d)
        """
        frames = {}

        # High-vol regime based on VIX level
        frames["high_vol_regime"] = (self.df_raw[VIX_COL] > self.vix_high_threshold).astype(int)

        # Bear market: SPX below its long-term moving average
        spx_ma = self.df_raw[SPX_COL].rolling(self.bear_ma_window).mean()
        frames["bear_market_flag"] = (self.df_raw[SPX_COL] < spx_ma).astype(int)

        # Vol spike: realized vol of SPX in top decile (rolling 252-day percentile rank)
        spx_realized_vol = log_returns[f"{SPX_COL}_log_ret"].rolling(WINDOW_SHORT).std() * np.sqrt(TRADING_DAYS)
        frames["vol_spike_flag"] = (
            spx_realized_vol.rolling(TRADING_DAYS)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] >= 0.90, raw=False)
            # .astype(int)
        )

        return pd.DataFrame(frames, index=log_returns.index)

    # ── 7. Descriptive statistics & validation ───────────────────────────────

    def descriptive_stats(self, log_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Summary statistics for log returns including Jarque-Bera normality test.
        Returns a DataFrame with mean, std, skewness, kurtosis, JB stat, JB p-value.
        """
        records = []
        for col in log_returns.columns:
            series = log_returns[col].dropna()
            jb_stat, jb_pval = stats.jarque_bera(series)
            records.append({
                "asset"    : col.replace("_log_ret", ""),
                "mean"     : series.mean(),
                "std"      : series.std(),
                "skewness" : series.skew(),
                "kurtosis" : series.kurt(),
                "jb_stat"  : jb_stat,
                "jb_pval"  : jb_pval,
                "normal"   : jb_pval > 0.05,   # True = cannot reject normality
            })
        return pd.DataFrame(records).set_index("asset")

    def validate_features(self, df_features: pd.DataFrame) -> None:
        """
        Basic sanity checks on the feature dataframe.
        Prints a warning for columns with excessive NaNs or implausible values.
        """
        print("\n── Feature validation ──────────────────────────────────")
        print(f"Shape: {df_features.shape}")

        nan_pct = df_features.isna().mean() * 100
        high_nan = nan_pct[nan_pct > 20]
        if not high_nan.empty:
            print(f"\nWARNING – columns with >20% NaN:\n{high_nan.round(1)}")
        else:
            print("NaN check: OK (no column exceeds 20% missing)")

        # Check that log returns are within a plausible daily range
        ret_cols = [c for c in df_features.columns if c.endswith("_log_ret")]
        for col in ret_cols:
            col_max = df_features[col].abs().max()
            if col_max > 0.5:
                print(f"WARNING – {col} has |log_ret| > 50% on at least one day ({col_max:.2%})")

        print("────────────────────────────────────────────────────────\n")

    # ── 8. Master builder ────────────────────────────────────────────────────

    def build(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline and return the enriched dataframe.

        Steps
        -----
        1. Log returns
        2. Rolling volatility (short / long / realized)
        3. Rolling momentum (short / long)
        4. Rolling SPX/VIX correlation
        5. Drawdown per asset
        6. Baseline regime labels (rule-based)
        7. Validation & descriptive stats (if verbose=True)
        """
        print("Building features...")

        log_returns  = self.compute_log_returns()
        rolling_vol  = self.compute_rolling_volatility(log_returns)
        rolling_mom  = self.compute_rolling_momentum(log_returns)
        spx_vix_corr = self.compute_spx_vix_correlation(log_returns)
        drawdowns    = self.compute_drawdown(log_returns)
        regime_flags = self.compute_baseline_regime_labels(log_returns)

        df_features = pd.concat(
            [self.df_raw[PRICE_COLS + [VIX_COL]], log_returns, rolling_vol, rolling_mom, spx_vix_corr, drawdowns, regime_flags],
            axis=1,
        )

        self.validate_features(df_features)

        if verbose:
            print("\n── Descriptive statistics (log returns) ────────────────")
            desc = self.descriptive_stats(log_returns)
            print(desc.round(4).to_string())
            print("\n── Correlation matrix (log returns) ────────────────────")
            print(log_returns.corr().round(3).to_string())

        self.df_features = df_features
        print(f"\nDone. Feature dataframe shape: {df_features.shape}")
        return df_features


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load raw price CSV produced by 01_ingest.py."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def save_features(df_features: pd.DataFrame, path: str) -> None:
    """Save the feature dataframe to parquet """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_features.to_parquet(path, index=True)
    print(f"Features saved to: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    raw_path     = os.path.join(cfg_path.workspace_root, cfg_path.raw_data)
    output_path  = os.path.join(cfg_path.workspace_root, cfg_path.processed_data)

    df_raw      = load_raw(raw_path)
    engineer    = FeatureEngineer(df_raw)
    df_features = engineer.build(verbose=True)
    save_features(df_features, output_path)


if __name__ == "__main__":
    main()