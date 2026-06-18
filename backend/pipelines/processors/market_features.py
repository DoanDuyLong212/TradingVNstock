import numpy as np
import pandas as pd

class MarketRegimeFeatureBuilder:
    @staticmethod
    def build(market_df: pd.DataFrame) -> pd.DataFrame:
        df = market_df.copy()
        df = df.sort_values("Ngay").reset_index(drop=True)

        # =====================================================
        # BASIC RETURNS (STRICT)
        # =====================================================
        df["index_ret_1d"] = df["index_close"].pct_change().shift(1)
        df["index_ret_20d"] = df["index_close"].pct_change(20).shift(1)
        df["index_ret_60d"] = df["index_close"].pct_change(60).shift(1)

        # =====================================================
        # TREND FEATURES (STRICT)
        # =====================================================
        df["index_ma50"] = (
            df["index_close"]
            .rolling(50, min_periods=50)
            .mean()
            .shift(1)
        )

        df["index_ma200"] = (
            df["index_close"]
            .rolling(200, min_periods=200)
            .mean()
            .shift(1)
        )

        df["index_dist_ma50"] = (
            df["index_close"].shift(1) / df["index_ma50"] - 1
        )

        df["index_dist_ma200"] = (
            df["index_close"].shift(1) / df["index_ma200"] - 1
        )

        df["index_ma200_slope_20d_pct"] = (
            df["index_ma200"] / df["index_ma200"].shift(20) - 1
        )

        df["index_ma50_slope_20d_pct"] = (
            df["index_ma50"] / df["index_ma50"].shift(20) - 1
        )

        df["index_trend_strength"] = (
            df["index_dist_ma50"] - df["index_dist_ma200"]
        )

        df["index_trend_regime"] = np.sign(
            df["index_ma200_slope_20d_pct"]
        )

        # =====================================================
        # VOLATILITY REGIME (STRICT)
        # =====================================================
        df["index_vol_20"] = (
            df["index_ret_1d"]
            .rolling(20)
            .std()
            .shift(1)
            * np.sqrt(252)
        )

        df["index_vol_60"] = (
            df["index_ret_1d"]
            .rolling(60)
            .std()
            .shift(1)
            * np.sqrt(252)
        )

        df["index_vol_ratio"] = np.where(
            df["index_vol_60"] > 0,
            df["index_vol_20"] / df["index_vol_60"],
            np.nan
        )

        # =====================================================
        # TRUE ATR (STRICT)
        # =====================================================
        high_low = df["index_high"].shift(1) - df["index_low"].shift(1)

        high_prev_close = (
            df["index_high"].shift(1) - df["index_close"].shift(2)
        ).abs()

        low_prev_close = (
            df["index_low"].shift(1) - df["index_close"].shift(2)
        ).abs()

        true_range = pd.concat(
            [high_low, high_prev_close, low_prev_close],
            axis=1
        ).max(axis=1)

        df["index_ATR_20"] = (
            true_range
            .rolling(20)
            .mean()
            .shift(1)
        )

        df["index_ATR_20_pct"] = (
            df["index_ATR_20"] / df["index_close"].shift(1)
        )

        # =====================================================
        # RETURN SHOCK (STRICT)
        # =====================================================
        rolling_mean = (
            df["index_ret_1d"]
            .rolling(20)
            .mean()
            .shift(1)
        )

        rolling_std = (
            df["index_ret_1d"]
            .rolling(20)
            .std()
            .shift(1)
        )

        df["index_ret_zscore_20"] = (
            (df["index_ret_1d"] - rolling_mean) /
            (rolling_std + 1e-8)
        )

        df["index_abs_zscore_20"] = (
            df["index_ret_zscore_20"].abs()
        )

        # =====================================================
        # DRAWDOWN (STRICT)
        # =====================================================
        df["index_rolling_max_252"] = (
            df["index_close"]
            .rolling(252)
            .max()
            .shift(1)
        )

        df["index_drawdown_1y"] = (
            df["index_close"].shift(1) /
            df["index_rolling_max_252"] - 1
        )

        # =====================================================
        # MOMENTUM ACCELERATION
        # =====================================================
        df["index_momentum_accel"] = (
            df["index_ret_20d"] - df["index_ret_60d"]
        )

        # =====================================================
        # LIQUIDITY PROXY (STRICT)
        # =====================================================
        if "index_volume" in df.columns:
            df["index_volume_ma20"] = (
                df["index_volume"]
                .rolling(20)
                .mean()
                .shift(1)
            )

            df["index_volume_ratio"] = np.where(
                df["index_volume_ma20"] > 0,
                df["index_volume"].shift(1) /
                df["index_volume_ma20"],
                np.nan
            )

        return df

def add_market_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_label"] = 1

    # Downtrend
    downtrend_mask = (
        (df["index_drawdown_1y"] < -0.18) &
        (
            (df["index_ma200_slope_20d_pct"] < -0.006) |
            (df["index_dist_ma200"] < -0.04)
        )
    )
    df.loc[downtrend_mask, "market_label"] = 0

    # Uptrend
    uptrend_mask = (
        (df["index_drawdown_1y"] > -0.08) &
        (df["index_ma200_slope_20d_pct"] > 0.007) &
        (df["index_dist_ma200"] > 0.035) &
        (df["index_vol_ratio"] < 1.35)
    )
    df.loc[uptrend_mask, "market_label"] = 2

    return df

def build_market_features(market_df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
    """Build regime features and add market label. If shift is True, shifts regime columns by 1 day to prevent lookahead."""
    df_features = MarketRegimeFeatureBuilder.build(market_df)
    df_labeled = add_market_label(df_features)
    
    if shift:
        regime_cols = [c for c in df_labeled.columns if c not in ["Ngay", "ngay"]]
        df_labeled[regime_cols] = df_labeled[regime_cols].shift(1)
        
    return df_labeled
