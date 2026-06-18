import json
import pandas as pd
from upstash_redis import Redis
from io import StringIO

def save_stock_features(client: Redis, stock_id: str, df: pd.DataFrame):
    """Serialize and save stock price history and features to Redis."""
    if df.empty:
        return
    
    # Copy to avoid modifying the original dataframe
    df_save = df.copy()
    
    # Ensure dates are string representation in ISO format
    if "Ngay" in df_save.columns:
        df_save["Ngay"] = df_save["Ngay"].dt.strftime("%Y-%m-%d")
    if "ngay" in df_save.columns:
        df_save["ngay"] = df_save["ngay"].dt.strftime("%Y-%m-%d")
        
    data_json = df_save.to_json(orient="records")
    client.set(f"stock:{stock_id}", data_json)

def load_stock_features(client: Redis, stock_id: str) -> pd.DataFrame:
    """Retrieve and deserialize stock price history and features from Redis."""
    raw_data = client.get(f"stock:{stock_id}")
    if not raw_data:
        return pd.DataFrame()
    
    df = pd.read_json(StringIO(raw_data), orient="records")
    if df.empty:
        return df
        
    if "Ngay" in df.columns:
        df["Ngay"] = pd.to_datetime(df["Ngay"])
    if "ngay" in df.columns:
        df["ngay"] = pd.to_datetime(df["ngay"])
        
    return df

def save_market_features(client: Redis, df: pd.DataFrame):
    """Serialize and save market features to Redis."""
    if df.empty:
        return
        
    df_save = df.copy()
    if "Ngay" in df_save.columns:
        df_save["Ngay"] = df_save["Ngay"].dt.strftime("%Y-%m-%d")
    if "ngay" in df_save.columns:
        df_save["ngay"] = df_save["ngay"].dt.strftime("%Y-%m-%d")
        
    data_json = df_save.to_json(orient="records")
    client.set("market:VNINDEX", data_json)

def load_market_features(client: Redis) -> pd.DataFrame:
    """Retrieve and deserialize market features from Redis."""
    raw_data = client.get("market:VNINDEX")
    if not raw_data:
        return pd.DataFrame()
        
    df = pd.read_json(raw_data, orient="records")
    if df.empty:
        return df
        
    if "Ngay" in df.columns:
        df["Ngay"] = pd.to_datetime(df["Ngay"])
    if "ngay" in df.columns:
        df["ngay"] = pd.to_datetime(df["ngay"])
        
    return df
