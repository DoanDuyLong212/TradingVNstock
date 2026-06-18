import logging
import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score


from backend.connectors.database_connection import get_engine
from backend.pipelines.processors.stock_loader import load_all_stocks
from backend.pipelines.processors.market_features import build_market_features
from backend.pipelines.processors.stock_features import BCDFeatureEngine, BCDEventEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RecoveryPointEngine:
    @staticmethod
    def build(df, recovery_window=20):
        df = df.copy()
        df["ngay"] = pd.to_datetime(df["ngay"])
        logs = []

        for stock, g in df.groupby("stock_id", sort=False):
            g = g.reset_index(drop=True)
            low = g["low"].to_numpy()
            close = g["close"].to_numpy()
            open_ = g["open"].to_numpy()
            ngay = g["ngay"].to_numpy()

            b_ngay = g["B_Ngay"].to_numpy()
            c_ngay = g["C_Ngay"].to_numpy()
            breakdown_ngay = g["breakdown_Ngay"].to_numpy()
            c_close_arr = g["C_close"].to_numpy()

            event_rows = np.flatnonzero(g["breakdown"].to_numpy() == 1)

            for idx in event_rows:
                c_close = c_close_arr[idx]
                if pd.isna(c_close):
                    continue

                end_idx = min(idx + recovery_window + 1, len(g))
                if idx + 1 >= end_idx:
                    continue

                future_lows = low[idx + 1:end_idx]
                if len(future_lows) == 0:
                    continue

                d_idx = idx + 1 + np.argmin(future_lows)
                d_low = low[d_idx]

                recovery_idx = d_idx + 1
                if recovery_idx >= len(g):
                    recovery_idx = -1

                logs.append({
                    "stock_id": stock,
                    "B_Ngay": b_ngay[idx],
                    "C_Ngay": c_ngay[idx],
                    "breakdown_Ngay": breakdown_ngay[idx],
                    "C_close": c_close,
                    "D_Ngay": ngay[d_idx],
                    "D_low": d_low,
                    "recovery_Ngay": ngay[recovery_idx] if recovery_idx >= 0 else pd.NaT,
                    "recovery_open": open_[recovery_idx] if recovery_idx >= 0 else np.nan,
                    "recovery_close": close[recovery_idx] if recovery_idx >= 0 else np.nan,
                    "recovery_found": int(recovery_idx >= 0)
                })

        return pd.DataFrame(logs)

class LabelEngine:
    @staticmethod
    def build(recovery_df, full_df, horizon=60, target_return=0.15):
        full_df = (
            full_df.copy()
            .sort_values(["stock_id", "ngay"])
            .reset_index(drop=True)
        )
        full_df["ngay"] = pd.to_datetime(full_df["ngay"])
        recovery_df = recovery_df.copy()
        recovery_df["recovery_Ngay"] = pd.to_datetime(recovery_df["recovery_Ngay"])

        stock_map = {}
        for stock, g in full_df.groupby("stock_id", sort=False):
            g = g.reset_index(drop=True)
            stock_map[stock] = {
                "dates": g["ngay"].to_numpy(dtype="datetime64[ns]"),
                "open": g["open"].to_numpy(dtype=float),
                "high": g["high"].to_numpy(dtype=float)
            }

        results = []
        for stock, events in recovery_df.groupby("stock_id", sort=False):
            stock_data = stock_map.get(stock)
            if stock_data is None:
                continue

            dates = stock_data["dates"]
            opens = stock_data["open"]
            highs = stock_data["high"]

            for event in events.itertuples(index=False):
                if pd.isna(event.recovery_Ngay):
                    continue

                recovery_date = np.datetime64(event.recovery_Ngay)
                pos = dates.searchsorted(recovery_date)

                if pos >= len(dates) or dates[pos] != recovery_date:
                    continue

                entry_idx = pos
                entry_price = opens[entry_idx]

                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                future_highs = highs[entry_idx + 1 : entry_idx + horizon + 1]
                if future_highs.size == 0 or np.all(np.isnan(future_highs)):
                    continue

                max_high = float(np.nanmax(future_highs))
                future_return = (max_high / entry_price) - 1
                label = int(future_return >= target_return)

                row = event._asdict()
                row.update({
                    "entry_Ngay": pd.Timestamp(dates[entry_idx]),
                    "open_entry": entry_price,
                    "max_high_60d": max_high,
                    "future_return": future_return,
                    "label": label
                })
                results.append(row)

        return pd.DataFrame(results)

def run_training():
    engine = get_engine()
    
    logger.info("Loading stock data from PostgreSQL...")
    df = load_all_stocks(engine)
    
    logger.info("Running BCD Feature Engineering...")
    mid_df = BCDFeatureEngine.build(df, lookback_peak=60, ma_ma=200)
    
    logger.info("Detecting B-C-D Events...")
    event_df = BCDEventEngine.build(
        mid_df,
        lookback=20,
        drop_pct=0.15,
        confirm_days=2,
        max_bc_days=20,
        max_cd_days=20
    )
    
    logger.info("Running RecoveryPoint Engine...")
    recovery_df = RecoveryPointEngine.build(event_df)
    
    logger.info("Building labels...")
    label_df = LabelEngine.build(
        recovery_df,
        event_df,
        horizon=60,
        target_return=0.15
    )
    
    breakdown_df = event_df[event_df["breakdown"] == 1].copy()
    logger.info(f"Detected breakdown events: {len(breakdown_df)}")
    
    train_df = breakdown_df.merge(
        label_df[
            [
                "stock_id",
                "breakdown_Ngay",
                "label",
                "future_return",
                "recovery_Ngay",
                "D_Ngay"
            ]
        ],
        on=["stock_id", "breakdown_Ngay"],
        how="inner"
    )
    
    logger.info(f"Merge train set length: {len(train_df)}")
    
    feature_cols = [
        "ret_20","ret_60","volatility_5","volatility_20",
        "distance_to_peak","distance_to_ma200",
        "ma_vol_5","ma_vol_20","volume_z",
        "ATR_compression_ratio",
        "bc_return","bc_days","cd_days",
        "peak_to_b","peak_to_c",
        "breakdown_strength"
    ]
    
    train_df["ngay"] = pd.to_datetime(train_df["ngay"])
    
    # Train / Val / Test Splits
    SPLIT_DATE_TRAIN = "2024-01-01"
    SPLIT_DATE_VAL   = "2025-01-01"
    
    train_set = train_df[train_df["ngay"] < pd.to_datetime(SPLIT_DATE_TRAIN)].copy()
    val_set   = train_df[
        (train_df["ngay"] >= pd.to_datetime(SPLIT_DATE_TRAIN)) &
        (train_df["ngay"] <  pd.to_datetime(SPLIT_DATE_VAL))
    ].copy()
    test_set  = train_df[train_df["ngay"] >= pd.to_datetime(SPLIT_DATE_VAL)].copy()
    
    logger.info(f"Split sizes: Train={train_set.shape[0]}, Val={val_set.shape[0]}, Test={test_set.shape[0]}")
    
    X_train = train_set[feature_cols]
    y_train = train_set["label"]
    X_val = val_set[feature_cols]
    y_val = val_set["label"]
    X_train = train_set[feature_cols].copy()
    y_train = train_set["label"]
    X_val = val_set[feature_cols].copy()
    y_val = val_set["label"]
    X_test = test_set[feature_cols].copy()
    y_test = test_set["label"]
    
    # Calculate medians on training set
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median()
    
    # Fill NaNs
    for col in feature_cols:
        X_train[col] = X_train[col].fillna(medians[col])
        X_val[col] = X_val[col].fillna(medians[col])
        X_test[col] = X_test[col].fillna(medians[col])
    
    # Compute scale_pos_weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / (n_pos + 1e-9)
    logger.info(f"scale_pos_weight: {scale_pos_weight}")
    
    # Train Model
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Fitting LightGBM Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50)],
    )
    
    # Calculate best precision threshold on Val set
    proba_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    best_prec, best_prec_thr = 0.0, 0.8 # Fallback to 0.8 if needed
    
    for t in thresholds:
        pred = (proba_val >= t).astype(int)
        prec = precision_score(y_val, pred, zero_division=0)
        if prec > best_prec:
            best_prec, best_prec_thr = prec, t
            
    logger.info(f"Best Precision on Val Set: {best_prec:.4f} at threshold: {best_prec_thr:.4f}")
    
    # Save model and meta parameters
    os.makedirs("backend/Stock_Filter/models", exist_ok=True)
    model_data = {
        "model": model,
        "features": feature_cols,
        "best_threshold": best_prec_thr,
        "medians": medians.to_dict()
    }
    
    with open("backend/Stock_Filter/models/model_1.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    logger.info("Model 1 successfully trained and saved to models/model_1.pkl")

if __name__ == "__main__":
    run_training()
