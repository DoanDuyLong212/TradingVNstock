import pandas as pd
from sqlalchemy import text

def load_all_market(engine, market_id: str = "VNINDEX") -> pd.DataFrame:
    """Load all market OHLCV data from the database and rename columns for the feature builders."""
    query = """
        SELECT ngay, open, high, low, close
        FROM market_ohlcv
        WHERE market_id = :market_id
        ORDER BY ngay
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"market_id": market_id})
    df["ngay"] = pd.to_datetime(df["ngay"])
    df = df.rename(columns={
        "ngay": "Ngay",
        "open": "index_open",
        "high": "index_high",
        "low": "index_low",
        "close": "index_close"
    })
    return df

def load_market_latest(engine, market_id: str = "VNINDEX", limit: int = 300) -> pd.DataFrame:
    """Load the latest N days of market OHLCV data."""
    query = """
        SELECT ngay, open, high, low, close
        FROM (
            SELECT ngay, open, high, low, close
            FROM market_ohlcv
            WHERE market_id = :market_id
            ORDER BY ngay DESC
            LIMIT :limit
        ) sub
        ORDER BY ngay ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"market_id": market_id, "limit": limit})
    df["ngay"] = pd.to_datetime(df["ngay"])
    df = df.rename(columns={
        "ngay": "Ngay",
        "open": "index_open",
        "high": "index_high",
        "low": "index_low",
        "close": "index_close"
    })
    return df
