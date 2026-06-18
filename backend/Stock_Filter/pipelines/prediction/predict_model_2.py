import logging
import os
import sys
import pickle
import pandas as pd
import numpy as np


from backend.Stock_Filter.redis_cache.redis_client import get_redis_client
from backend.Stock_Filter.redis_cache.feature_store import load_stock_features

logger = logging.getLogger(__name__)

def predict_model_2() -> pd.DataFrame:
    """Run inference for Model 2 (SEPA/Breakout model) on today's data."""
    model_path = "backend/Stock_Filter/models/model_2.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model 2 not found at {model_path}. Please run training script first.")
        return pd.DataFrame()
        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    model = model_data["model"]
    feature_cols = model_data["features"]
    threshold = model_data["best_threshold"]
    medians = model_data.get("medians", {})
    
    redis_client = get_redis_client()
    
    # Get all keys matching stock:*
    keys = redis_client.keys("stock:*")
    stock_ids = [k.split(":")[1] for k in keys]

    logger.info(f"Found {len(keys)} Redis keys")
    logger.info(f"Sample keys: {keys[:5]}")
    
    signals = []
    
    for stock_id in stock_ids:
        try:
            df = load_stock_features(redis_client, stock_id)
            if df.empty:
                continue
                
            # Sort by date and check the latest row
            df = df.sort_values("ngay").reset_index(drop=True)
            latest_row = df.iloc[-1]
            
            # Model 2 triggers only on Breakout days
            if latest_row.get("is_breakout") == 1:
                # Apply SEPA hard filters on the latest row
                # cond_dist = latest_row.get("distance_from_52w_high", -999.0) > -0.35
                # cond_rs = latest_row.get("RS_percentile_60d", -999.0) > 0.5
                # cond_depth = latest_row.get("base_depth_percent", -999.0) > -0.4
                # cond_vol = latest_row.get("volatility_compression_ratio", 999.0) < 0.95
                
                # if cond_dist and cond_rs and cond_depth and cond_vol:
                    # Prepare features
                feat_dict = {}
                for col in feature_cols:
                    val = latest_row.get(col, np.nan)
                    # Handle inf values
                    if val == np.inf or val == -np.inf:
                        val = np.nan
                    # Fill with training median if NaN
                    if pd.isna(val):
                        val = medians.get(col, 0.0)
                    feat_dict[col] = [val]
                    
                X_pred = pd.DataFrame(feat_dict)
                proba = model.predict_proba(X_pred)[0, 1]
                
                logger.info(f"Model 2 prediction for {stock_id} on {latest_row['ngay'].date()}: proba={proba:.4f} (threshold={threshold:.4f})")
                
                if proba >= threshold:
                    signals.append({
                        "stock_id": stock_id,
                        "ngay": latest_row["ngay"],
                        "model": "Breakout_SEPA_Model_2",
                        "probability": proba,
                        "close": latest_row.get("close"),
                        "volume": latest_row.get("volume"),
                        "base_length": latest_row.get("base_length")
                    })
        except Exception as e:
            logger.error(f"Error running model 2 inference on {stock_id}: {e}")
            
    if not signals:
        return pd.DataFrame()
        
    return pd.DataFrame(signals)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_signals = predict_model_2()
    if not df_signals.empty:
        print("Model 2 Signals:")
        print(df_signals.to_string())
    else:
        print("No Model 2 signals today.")
