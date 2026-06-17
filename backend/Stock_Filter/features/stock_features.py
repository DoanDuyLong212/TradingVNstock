import numpy as np
import pandas as pd

class BCDFeatureEngine:
    @staticmethod
    def build(df: pd.DataFrame, lookback_peak=60, ma_ma=200) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["stock_id", "ngay"]).reset_index(drop=True)

        def compute_features(g):
            stock_id = g.name
            g = g.copy().reset_index(drop=True)
            g["stock_id"] = stock_id

            # =========================
            # PRICE RETURNS
            # =========================
            g["ret_20"] = g["close"].pct_change(20).shift(1)
            g["ret_60"] = g["close"].pct_change(60).shift(1)

            # =========================
            # VOLATILITY
            # =========================
            g["volatility_5"] = g["close"].pct_change().rolling(5).std().shift(1)
            g["volatility_20"] = g["close"].pct_change().rolling(20).std().shift(1)

            # =========================
            # DISTANCE TO PEAK
            # =========================
            rolling_peak = g["close"].rolling(lookback_peak).max().shift(1)
            g["distance_to_peak"] = (g["close"] / rolling_peak - 1)

            # =========================
            # MA200 DISTANCE
            # =========================
            ma200 = g["close"].rolling(ma_ma).mean().shift(1)
            g["distance_to_ma200"] = (g["close"] / ma200 - 1)

            # =========================
            # VOLUME FEATURES
            # =========================
            g["ma_vol_5"] = g["volume"].rolling(5).mean().shift(1)
            g["ma_vol_20"] = g["volume"].rolling(20).mean().shift(1)

            g["volume_z"] = (
                (g["volume"] - g["ma_vol_20"]) /
                (g["volume"].rolling(20).std().shift(1) + 1e-9)
            )

            # =========================
            # ATR COMPRESSION RATIO
            # =========================
            high_low = g["high"] - g["low"]
            high_close = (g["high"] - g["close"].shift()).abs()
            low_close = (g["low"] - g["close"].shift()).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            atr5 = tr.rolling(5).mean()
            atr20 = tr.rolling(20).mean()

            g["ATR_compression_ratio"] = (atr5 / (atr20 + 1e-9)).shift(1)

            # BCD structure features (only if B_close/C_close present)
            if "B_close" in g.columns and "C_close" in g.columns:
                g["b_to_c"] = (g["C_close"] / g["B_close"] - 1)
                g["b_rebound"] = np.nan
                g["c_rebound"] = np.nan

                for i in range(len(g)):
                    if not pd.isna(g.loc[i, "B_close"]):
                        window_b = g.loc[max(0, i-5):i]
                        g.loc[i, "b_rebound"] = (
                            window_b["close"].max() / g.loc[i, "B_close"] - 1
                        )
                    if not pd.isna(g.loc[i, "C_close"]):
                        window_c = g.loc[max(0, i-5):i]
                        g.loc[i, "c_rebound"] = (
                            window_c["close"].max() / g.loc[i, "C_close"] - 1
                        )

            return g

        df = df.groupby("stock_id", group_keys=False).apply(compute_features)
        return df

class BCDEventEngine:
    @staticmethod
    def build(
        df: pd.DataFrame,
        lookback=20,
        drop_pct=0.20,
        confirm_days=3,
        max_bc_days=20,
        max_cd_days=20
    ) -> pd.DataFrame:
        def is_lowest_in_lookback(arr, idx, lookback):
            start = max(0, idx - lookback + 1)
            return idx == start + np.argmin(arr[start:idx + 1])

        df = (
            df.copy()
            .sort_values(["stock_id", "ngay"])
            .reset_index(drop=True)
        )

        df["ngay"] = pd.to_datetime(df["ngay"])

        # event flags
        df["bcdevent"] = 0
        df["breakdown"] = 0

        # debug / event columns
        df["B_Ngay"] = pd.NaT
        df["C_Ngay"] = pd.NaT
        df["breakdown_Ngay"] = pd.NaT

        df["B_idx"] = np.nan
        df["C_idx"] = np.nan
        df["breakdown_idx"] = np.nan

        df["peak_before_B"] = np.nan
        df["peak_before_C"] = np.nan

        df["B_close"] = np.nan
        df["C_close"] = np.nan
        df["breakdown_close"] = np.nan

        df["bc_return"] = np.nan
        df["bc_days"] = np.nan
        df["cd_days"] = np.nan
        df["peak_to_b"] = np.nan
        df["peak_to_c"] = np.nan
        df["breakdown_strength"] = np.nan

        for stock, g in df.groupby("stock_id", sort=False):
            g = g.reset_index()

            orig_idx = g["index"].to_numpy()
            close = g["close"].to_numpy(dtype=float)
            ngay = g["ngay"].to_numpy()

            n = len(g)
            i = lookback

            while i < n:
                # 1) DROP > drop_pct FROM PEAK
                start = max(0, i - lookback)
                peak_price = close[start:i + 1].max()
                drop_real = (close[i] / peak_price) - 1

                if drop_real > -drop_pct:
                    i += 1
                    continue

                # 2) FIND B
                B_idx = i
                B_close = close[i]
                last_B_update = i

                j = i + 1
                while j < n:
                    cj = close[j]
                    if cj < B_close:
                        B_close = cj
                        B_idx = j
                        last_B_update = j
                    if j - last_B_update >= confirm_days:
                        break
                    j += 1

                if last_B_update + confirm_days >= n:
                    i += 1
                    continue

                if not is_lowest_in_lookback(close, B_idx, lookback):
                    i += 1
                    continue

                peak_before_B = close[max(0, B_idx - lookback):B_idx + 1].max()

                # 3) FIND C
                C_idx = None
                C_close = None
                last_C_update = None

                k = B_idx + confirm_days + 1
                upper_bc = min(B_idx + max_bc_days + 1, n)

                while k < upper_bc:
                    ck = close[k]
                    if ck < B_close:
                        C_idx = k
                        C_close = ck
                        last_C_update = k

                        m = k + 1
                        while m < upper_bc:
                            cm = close[m]
                            if cm < C_close:
                                C_close = cm
                                C_idx = m
                                last_C_update = m
                            if m - last_C_update >= confirm_days:
                                break
                            m += 1
                        break
                    k += 1

                if C_idx is None:
                    i += 1
                    continue

                if not is_lowest_in_lookback(close, C_idx, lookback):
                    i += 1
                    continue

                peak_before_C = close[max(0, C_idx - lookback):C_idx + 1].max()

                # 4) FIND BREAKDOWN
                breakdown_idx = None
                k = C_idx + confirm_days + 1
                upper_cd = min(C_idx + max_cd_days + 1, n)

                while k < upper_cd:
                    if close[k] < C_close:
                        breakdown_idx = k
                        break
                    k += 1

                if breakdown_idx is None:
                    i += 1
                    continue

                breakdown_close = close[breakdown_idx]

                # SAVE EVENT
                oidx = orig_idx[breakdown_idx]
                df.at[oidx, "bcdevent"] = 1
                df.at[oidx, "breakdown"] = 1

                df.at[oidx, "B_Ngay"] = ngay[B_idx]
                df.at[oidx, "C_Ngay"] = ngay[C_idx]
                df.at[oidx, "breakdown_Ngay"] = ngay[breakdown_idx]

                df.at[oidx, "B_idx"] = B_idx
                df.at[oidx, "C_idx"] = C_idx
                df.at[oidx, "breakdown_idx"] = breakdown_idx

                df.at[oidx, "peak_before_B"] = peak_before_B
                df.at[oidx, "peak_before_C"] = peak_before_C

                df.at[oidx, "B_close"] = B_close
                df.at[oidx, "C_close"] = C_close
                df.at[oidx, "breakdown_close"] = breakdown_close

                df.at[oidx, "bc_return"] = (C_close / B_close) - 1
                df.at[oidx, "bc_days"] = C_idx - B_idx
                df.at[oidx, "cd_days"] = breakdown_idx - C_idx
                df.at[oidx, "peak_to_b"] = (B_close / peak_before_B) - 1
                df.at[oidx, "peak_to_c"] = (C_close / peak_before_C) - 1
                df.at[oidx, "breakdown_strength"] = (breakdown_close / C_close) - 1

                i = breakdown_idx + 1

        return df

class BreakoutEventEngine:
    @staticmethod
    def build(
        df: pd.DataFrame,
        breakout_lookback=60,
        min_base_length=20,
        max_base_length=120,
        breakout_buffer=0.012,
        cooldown_days=20,
        min_volume_ratio=1.2,
        max_atr_pct=0.08
    ) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["stock_id", "Ngay"]).reset_index(drop=True)

        results = []

        for stock, g in df.groupby("stock_id"):
            g = g.copy().reset_index(drop=True)
            last_breakout_idx = -999

            # Rolling High
            g["rolling_high"] = (
                g["close"]
                .rolling(breakout_lookback, min_periods=breakout_lookback)
                .max()
                .shift(1)
            )

            # ATR
            high_low = g["high"] - g["low"]
            high_prev_close = (g["high"] - g["close"].shift()).abs()
            low_prev_close = (g["low"] - g["close"].shift()).abs()

            g["true_range"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
            g["ATR_20"] = g["true_range"].rolling(20).mean().shift(1)
            g["ATR_pct"] = g["ATR_20"] / g["close"].shift(1)

            # Volume
            if "volume" in g.columns:
                g["vol_ma20"] = g["volume"].rolling(20).mean().shift(1)
                g["vol_ma50"] = g["volume"].rolling(50).mean().shift(1)
                g["breakout_volume_ratio"] = (
                    g["volume"] / g["vol_ma20"]
                )

            # Initialize event columns
            g["is_breakout"] = 0
            g["base_length"] = np.nan
            g["base_depth"] = np.nan
            g["base_return"] = np.nan
            g["contraction_ratio"] = np.nan
            g["tight_range_15"] = np.nan
            g["base_atr_mean"] = np.nan
            g["base_volatility"] = np.nan
            g["vol_dryup_ratio"] = np.nan

            g["breakout_strength_atr"] = np.nan
            g["breakout_range"] = np.nan
            g["close_strength"] = np.nan

            g["ret_20d_pre"] = np.nan
            g["ret_60d_pre"] = np.nan
            g["ret_1y_pre"] = np.nan

            # Main detection loop
            for i in range(breakout_lookback, len(g)):
                if i - last_breakout_idx < cooldown_days:
                    continue

                if pd.isna(g.loc[i, "rolling_high"]):
                    continue

                breakout_level = g.loc[i, "rolling_high"] * (1 + breakout_buffer)

                if g.loc[i, "high"] <= breakout_level:
                    continue

                if g.loc[i, "close"] <= breakout_level:
                    continue

                if g.loc[i, "ATR_pct"] > max_atr_pct:
                    continue

                if "volume" in g.columns:
                    if g.loc[i, "breakout_volume_ratio"] < min_volume_ratio:
                        continue

                # Base extraction
                best_score = -np.inf
                best_metrics = None
                best_len = None

                for base_len in range(min_base_length, max_base_length + 1):
                    if i - base_len < 0:
                        break

                    base = g.iloc[i-base_len:i]
                    base_high = base["high"].max()
                    base_low = base["low"].min()
                    base_depth = (base_high - base_low) / base_high
                    base_return = (
                        base["close"].iloc[-1] /
                        base["close"].iloc[0] - 1
                    )

                    atr_series = base["ATR_pct"].dropna()
                    if len(atr_series) < 10:
                        continue

                    first_half = atr_series.iloc[:len(atr_series)//2].mean()
                    second_half = atr_series.iloc[len(atr_series)//2:].mean()

                    if first_half <= 0:
                        continue

                    contraction_ratio = second_half / first_half
                    tight_window = base.iloc[-15:]

                    tight_range = (
                        tight_window["high"].max() -
                        tight_window["low"].min()
                    ) / tight_window["high"].max()

                    base_volatility = base["close"].pct_change().std()
                    base_atr_mean = base["ATR_pct"].mean()

                    if "volume" in base.columns:
                        first_vol = base["volume"].iloc[:base_len//2].mean()
                        last_vol = base["volume"].iloc[base_len//2:].mean()
                        vol_dryup = last_vol / first_vol if first_vol > 0 else np.nan
                    else:
                        vol_dryup = np.nan

                    length_penalty = 0.002 * base_len
                    score = (
                        -base_depth
                        -contraction_ratio
                        -tight_range
                        + (1 - vol_dryup if not np.isnan(vol_dryup) else 0)
                        + length_penalty
                    )

                    if score > best_score:
                        best_score = score
                        best_len = base_len
                        best_metrics = (
                            base_depth,
                            base_return,
                            contraction_ratio,
                            tight_range,
                            base_volatility,
                            base_atr_mean,
                            vol_dryup
                        )

                if best_metrics is None:
                    continue

                # Breakout features
                breakout_strength = np.nan
                if g.loc[i, "ATR_20"] > 0:
                    breakout_strength = (
                        g.loc[i, "close"] -
                        g.loc[i, "rolling_high"]
                    ) / g.loc[i, "ATR_20"]

                breakout_range = (
                    g.loc[i, "high"] -
                    g.loc[i, "low"]
                ) / g.loc[i, "close"]

                close_strength = (
                    g.loc[i, "close"] -
                    g.loc[i, "low"]
                ) / (
                    g.loc[i, "high"] -
                    g.loc[i, "low"] + 1e-9
                )

                # Pre-breakout momentum
                ret20 = g.loc[i-1, "close"] / g.loc[i-21, "close"] - 1 if i >= 21 else np.nan
                ret60 = g.loc[i-1, "close"] / g.loc[i-61, "close"] - 1 if i >= 61 else np.nan
                ret1y = g.loc[i-1, "close"] / g.loc[i-253, "close"] - 1 if i >= 253 else np.nan

                # Register event
                g.loc[i, "is_breakout"] = 1
                last_breakout_idx = i

                g.loc[i, "base_length"] = best_len
                g.loc[i, "base_depth"] = best_metrics[0]
                g.loc[i, "base_return"] = best_metrics[1]
                g.loc[i, "contraction_ratio"] = best_metrics[2]
                g.loc[i, "tight_range_15"] = best_metrics[3]
                g.loc[i, "base_volatility"] = best_metrics[4]
                g.loc[i, "base_atr_mean"] = best_metrics[5]
                g.loc[i, "vol_dryup_ratio"] = best_metrics[6]

                g.loc[i, "breakout_strength_atr"] = breakout_strength
                g.loc[i, "breakout_range"] = breakout_range
                g.loc[i, "close_strength"] = close_strength

                g.loc[i, "ret_20d_pre"] = ret20
                g.loc[i, "ret_60d_pre"] = ret60
                g.loc[i, "ret_1y_pre"] = ret1y

            results.append(g)

        return pd.concat(results).reset_index(drop=True)

def build_features_sepa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["stock_id", "Ngay"]).reset_index(drop=True)

    # 1) TREND ALIGNMENT
    df["ma50"] = df.groupby("stock_id")["close"] \
        .transform(lambda x: x.rolling(50, min_periods=50).mean().shift(1))
    df["ma150"] = df.groupby("stock_id")["close"] \
        .transform(lambda x: x.rolling(150, min_periods=150).mean().shift(1))
    df["ma200"] = df.groupby("stock_id")["close"] \
        .transform(lambda x: x.rolling(200, min_periods=200).mean().shift(1))

    df["ma50_ma150_ratio"] = df["ma50"] / df["ma150"]
    df["ma150_ma200_ratio"] = df["ma150"] / df["ma200"]
    df["price_ma50_ratio"] = df.groupby("stock_id")["close"].shift(1) / df["ma50"]

    # 2) 52W HIGH + RELATIVE STRENGTH
    rolling_252_high = (
        df.groupby("stock_id")["close"]
        .transform(lambda x: x.rolling(252, min_periods=252).max().shift(1))
    )
    df["distance_from_52w_high"] = df["close"].shift(1) / rolling_252_high - 1

    df["ret_60d"] = df.groupby("stock_id")["close"] \
        .transform(lambda x: x.pct_change(60).shift(1))
    df["RS_percentile_60d"] = df.groupby("Ngay")["ret_60d"].rank(pct=True)

    # RS NEW HIGH
    rs_line = df["close"] / df["index_close"]
    rs_120_high = (
        rs_line.groupby(df["stock_id"])
        .transform(lambda x: x.rolling(120, min_periods=120).max().shift(1))
    )
    df["RS_new_high"] = (rs_line > rs_120_high).astype(int)

    # 3) BASE STRUCTURE
    prev_close = df.groupby("stock_id")["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - prev_close),
            abs(df["low"] - prev_close)
        )
    )

    df["atr_14"] = df.groupby("stock_id")["tr"] \
        .transform(lambda x: x.rolling(14, min_periods=14).mean().shift(1))
    df["atr_60"] = df.groupby("stock_id")["atr_14"] \
        .transform(lambda x: x.rolling(60, min_periods=60).mean().shift(1))

    df["volatility_compression_ratio"] = df["atr_14"] / df["atr_60"]

    rolling_low_60 = (
        df.groupby("stock_id")["low"]
        .transform(lambda x: x.rolling(60, min_periods=60).min().shift(1))
    )
    rolling_high_60 = (
        df.groupby("stock_id")["high"]
        .transform(lambda x: x.rolling(60, min_periods=60).max().shift(1))
    )
    df["base_depth_percent"] = rolling_low_60 / rolling_high_60 - 1

    df["price_tightness_20"] = (
        df.groupby("stock_id")["close"]
        .transform(lambda x: x.rolling(20, min_periods=20).std().shift(1))
    )

    # 4) VOLUME
    df["volume_ma20"] = df.groupby("stock_id")["volume"] \
        .transform(lambda x: x.rolling(20, min_periods=20).mean().shift(1))
    df["volume_ma50"] = df.groupby("stock_id")["volume"] \
        .transform(lambda x: x.rolling(50, min_periods=50).mean().shift(1))

    # 5) BREAKOUT CONFIRMATION
    rolling_high_20 = (
        df.groupby("stock_id")["high"]
        .transform(lambda x: x.rolling(20, min_periods=20).max().shift(1))
    )
    df["breakout_strength"] = df["close"].shift(1) / rolling_high_20 - 1
    df["breakout_volume_ratio"] = (
        df["volume"] / (df["volume_ma20"] + 1e-6)
    )

    # 6) RESISTANCE PRESSURE
    rolling_high_30 = (
        df.groupby("stock_id")["high"]
        .transform(lambda x: x.rolling(30, min_periods=30).max().shift(1))
    )
    df["near_resistance"] = (
        df["close"].shift(1) >= rolling_high_30 * 0.97
    ).astype(int)

    df["resistance_tests_30d"] = (
        df.groupby("stock_id")["near_resistance"]
        .transform(lambda x: x.rolling(30, min_periods=5).sum().shift(1))
    )

    return df

def apply_sepa_hard_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    conditions = (
        (df["distance_from_52w_high"] > -0.35) &
        (df["RS_percentile_60d"] > 0.5) &
        (df["base_depth_percent"] > -0.4) &
        (df["volatility_compression_ratio"] < 0.95)
    )
    return df[conditions].copy()
