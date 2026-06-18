import logging
import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text

# Add root directory to sys.path to resolve data_access and features imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.mcp_data.connectors.database_connection import get_engine
from backend.mcp_data.processors.stock_loader import load_all_stocks
from backend.mcp_data.processors.market_loader import load_all_market
from backend.Stock_Filter.features.market_features import build_market_features
from backend.Stock_Filter.features.stock_features import BreakoutEventEngine, build_features_sepa, apply_sepa_hard_filter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BreakoutLabelEngine:
    @staticmethod
    def build(
        df: pd.DataFrame,
        tp_atr: float = 8.0,
        sl_atr: float = 4.0,
        lookahead: int = 60,
        max_return_threshold: float = 0.15,
        stock_id_col: str = None,
        entry_mode: str = "next_open",
    ) -> pd.DataFrame:
        data = df.copy()
        sort_cols = ["Ngay"] if stock_id_col is None else [stock_id_col, "Ngay"]
        data = data.sort_values(sort_cols).reset_index(drop=True)

        data["label"] = np.nan
        data["entry_price"] = np.nan
        data["tp_price"] = np.nan
        data["sl_price"] = np.nan
        data["exit_price"] = np.nan
        data["exit_reason"] = None
        data["label_horizon_days"] = np.nan
        data["realized_return"] = np.nan
        data["max_return_lookahead"] = np.nan
        data["max_drawdown_lookahead"] = np.nan
        data["MFE"] = np.nan
        data["MAE"] = np.nan
        
        data["ret_5d"] = np.nan
        data["ret_10d"] = np.nan
        data["ret_20d"] = np.nan
        data["ret_60d"] = np.nan
        data["days_to_tp"] = np.nan
        data["days_to_sl"] = np.nan

        if stock_id_col is None:
            groups = [(None, data)]
        else:
            groups = data.groupby(stock_id_col, group_keys=False)

        for key, g in groups:
            g = g.reset_index()
            n = len(g)
            breakout_positions = g.index[g["is_breakout"] == 1]

            for pos in breakout_positions:
                if pos + 1 >= n:
                    continue

                if entry_mode == "close":
                    entry_price = g.loc[pos, "close"]
                    entry_pos = pos
                else:
                    entry_price = g.loc[pos + 1, "open"]
                    entry_pos = pos + 1

                atr = g.loc[pos, "ATR_20"]
                if pd.isna(atr) or atr <= 0:
                    continue

                tp_price = entry_price + tp_atr * atr
                sl_price = entry_price - sl_atr * atr
                original_idx = g.loc[pos, "index"]

                data.loc[original_idx, "entry_price"] = entry_price
                data.loc[original_idx, "tp_price"] = tp_price
                data.loc[original_idx, "sl_price"] = sl_price

                end_pos = min(entry_pos + lookahead, n - 1)
                future = g.loc[entry_pos:end_pos]

                if future.empty:
                    continue

                max_high = future["high"].max()
                min_low = future["low"].min()

                max_return = (max_high - entry_price) / entry_price
                max_drawdown = (min_low - entry_price) / entry_price

                data.loc[original_idx, "max_return_lookahead"] = max_return
                data.loc[original_idx, "max_drawdown_lookahead"] = max_drawdown
                data.loc[original_idx, "MFE"] = max_high - entry_price
                data.loc[original_idx, "MAE"] = entry_price - min_low

                label = 0
                horizon_days = lookahead
                exit_price = np.nan
                exit_reason = "timeout"
                days_to_tp = np.nan
                days_to_sl = np.nan

                for future_pos, row in future.iterrows():
                    hit_tp = row["high"] >= tp_price
                    hit_sl = row["low"] <= sl_price

                    if hit_sl and hit_tp:
                        label = 0
                        horizon_days = future_pos - entry_pos
                        exit_price = sl_price
                        exit_reason = "sl"
                        days_to_sl = horizon_days
                        break

                    if hit_sl:
                        label = 0
                        horizon_days = future_pos - entry_pos
                        exit_price = sl_price
                        exit_reason = "sl"
                        days_to_sl = horizon_days
                        break

                    if hit_tp:
                        label = 1
                        horizon_days = future_pos - entry_pos
                        exit_price = tp_price
                        exit_reason = "tp"
                        days_to_tp = horizon_days
                        break

                if exit_reason == "timeout":
                    exit_price = future.iloc[-1]["close"]
                    label = 1 if max_return >= max_return_threshold else 0

                realized_return = (exit_price - entry_price) / entry_price

                data.loc[original_idx, "label"] = label
                data.loc[original_idx, "label_horizon_days"] = horizon_days
                data.loc[original_idx, "exit_price"] = exit_price
                data.loc[original_idx, "exit_reason"] = exit_reason
                data.loc[original_idx, "realized_return"] = realized_return
                data.loc[original_idx, "days_to_tp"] = days_to_tp
                data.loc[original_idx, "days_to_sl"] = days_to_sl

                for h, col in [(5,"ret_5d"),(10,"ret_10d"),(20,"ret_20d"),(60,"ret_60d")]:
                    if entry_pos + h < n:
                        ret = (g.loc[entry_pos + h, "close"] / entry_price - 1)
                        data.loc[original_idx, col] = ret

        return data

def run_training():
    engine = get_engine()
    
    logger.info("Loading market data...")
    market_df = load_all_market(engine)
    
    logger.info("Building market regime features...")
    market_features = build_market_features(market_df, shift=True)
    
    logger.info("Loading stock data...")
    stock_df = load_all_stocks(engine)
    stock_df = stock_df.rename(columns={"ngay": "Ngay"})
    
    logger.info("Merging stock + market...")
    regime_cols = [c for c in market_features.columns if c not in ["Ngay", "ngay"]]
    full_df = (
        stock_df
        .merge(
            market_features[["Ngay"] + regime_cols],
            on="Ngay",
            how="left"
        )
        .sort_values(["stock_id", "Ngay"])
        .reset_index(drop=True)
    )
    
    logger.info("Detecting breakouts...")
    event_df = BreakoutEventEngine.build(
        df=full_df,
        breakout_lookback=100,
        min_base_length=12,
        max_base_length=120,
        cooldown_days=12
    )
    
    logger.info("Building breakout labels...")
    labeled_df = BreakoutLabelEngine.build(
        df=event_df,
        tp_atr=10.0,
        sl_atr=6.0,
        lookahead=120,
        stock_id_col="stock_id",
        entry_mode="next_open"
    )
    
    logger.info("Building SEPA features...")
    feature_df = build_features_sepa(labeled_df)
    
    TRADE_START = "2010-01-01"
    feature_df = feature_df[feature_df["Ngay"] >= pd.Timestamp(TRADE_START)].reset_index(drop=True)
    
    breakout_df = feature_df[
        (feature_df["is_breakout"] == 1) &
        (feature_df["label"].notna())
    ].reset_index(drop=True)
    
    logger.info(f"Breakout events: {len(breakout_df)}")
    
    logger.info("Applying SEPA hard filters...")
    filtered_df = apply_sepa_hard_filter(breakout_df)
    logger.info(f"After filter: {len(filtered_df)}")
    
    feature_cols = [
        "base_depth",
        "base_length",
        "base_return",
        "tight_range_15",
        "contraction_ratio",
        "breakout_volume_ratio",
        "vol_dryup_ratio",
        "base_volatility",
        "ret_20d_pre",
        "ret_60d_pre",
        "close_strength",
        "ATR_pct",
        "vol_ma20",
        "index_dist_ma200",
        "index_vol_ratio",
        "index_momentum_accel",
        "market_label"
    ]
    
    train_df = filtered_df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    
    SPLIT_DATE = pd.Timestamp("2024-01-01")
    train_mask = train_df["Ngay"] < SPLIT_DATE
    test_mask  = train_df["Ngay"] >= SPLIT_DATE
    
    X_train = train_df.loc[train_mask, feature_cols].copy()
    y_train = train_df.loc[train_mask, "label"].astype(int)
    
    # Feature Selection on X_train
    # 1. Missing ratio filter
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    missing_ratio = X_train.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
    X_train = X_train.drop(columns=high_missing_cols)
    
    # Fill remaining NaNs with median
    medians = X_train.median()
    for col in X_train.columns:
        X_train[col] = X_train[col].fillna(medians[col])
        
    # 2. Low variance filter
    low_var_cols = X_train.columns[X_train.var() < 1e-6].tolist()
    X_train = X_train.drop(columns=low_var_cols)
    
    # 3. High correlation filter
    protected = ["RS_percentile_60d", "volatility_compression_ratio"]
    corr_matrix = X_train.corr().abs()
    label_corr = X_train.corrwith(y_train).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > 0.85].tolist()
        for row in high_corr:
            if row in to_drop or col in to_drop:
                continue
            if row in protected or col in protected:
                continue
            if label_corr[row] >= label_corr[col]:
                to_drop.add(col)
            else:
                to_drop.add(row)
                
    X_train = X_train.drop(columns=list(to_drop))
    final_features = X_train.columns.tolist()
    
    # Re-calculate medians on the final features only to save them
    final_medians = medians[final_features].to_dict()
    
    logger.info(f"Final selected features ({len(final_features)}): {final_features}")
    
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-9)
    
    params = dict(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=24,
        max_depth=-1,
        min_child_samples=15,
        min_split_gain=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    
    logger.info("Training Model 2 (Breakout)...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X_train, y_train)
    
    os.makedirs("backend/Stock_Filter/models", exist_ok=True)
    model_data = {
        "model": final_model,
        "features": final_features,
        "best_threshold": 0.8,
        "high_missing_cols": high_missing_cols,
        "low_var_cols": low_var_cols,
        "dropped_corr_cols": list(to_drop),
        "medians": final_medians
    }
    
    with open("backend/Stock_Filter/models/model_2.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    logger.info("Model 2 successfully trained and saved to models/model_2.pkl")

if __name__ == "__main__":
    run_training()
