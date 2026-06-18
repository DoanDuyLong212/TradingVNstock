import pandas as pd
from sqlalchemy import text

def load_all_stocks(engine) -> pd.DataFrame:
    """Load all stock OHLCV data from the database."""
    query = """
        SELECT ngay, stock_id, open, high, low, close, volume
        FROM ohlcv
        ORDER BY stock_id, ngay
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df["ngay"] = pd.to_datetime(df["ngay"])
    return df

def load_stock_history(engine, stock_id: str) -> pd.DataFrame:
    """Load full OHLCV history for a specific stock."""
    query = """
        SELECT ngay, stock_id, open, high, low, close, volume
        FROM ohlcv
        WHERE stock_id = :stock_id
        ORDER BY ngay
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"stock_id": stock_id})
    df["ngay"] = pd.to_datetime(df["ngay"])
    return df

def load_stock_latest(engine, stock_id: str, limit: int = 300) -> pd.DataFrame:
    """Load the latest N days of OHLCV history for a specific stock."""
    query = """
        SELECT ngay, stock_id, open, high, low, close, volume
        FROM (
            SELECT ngay, stock_id, open, high, low, close, volume
            FROM ohlcv
            WHERE stock_id = :stock_id
            ORDER BY ngay DESC
            LIMIT :limit
        ) sub
        ORDER BY ngay ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"stock_id": stock_id, "limit": limit})
    df["ngay"] = pd.to_datetime(df["ngay"])
    return df
