import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from backend.pipelines.processors.feature_store import (
    load_stock_features, save_stock_features,
    load_market_features, save_market_features
)
from backend.pipelines.processors.market_features import build_market_features
from backend.pipelines.processors.stock_features import (
    BCDFeatureEngine, BCDEventEngine,
    BreakoutEventEngine, build_features_sepa
)

logger = logging.getLogger(__name__)

FEATURE_WINDOW = 350

def update_market_features_store(engine, redis_client) -> pd.DataFrame:
    """Load latest market data from database, compute features, and save to Redis."""
    logger.info("Updating market features...")
    
    # We load market data (at least 300 days)
    query = """
        SELECT ngay, open, high, low, close
        FROM market_ohlcv
        WHERE market_id = 'VNINDEX'
        ORDER BY ngay
    """
    with engine.connect() as conn:
        market_df = pd.read_sql(text(query), conn)
    
    if market_df.empty:
        logger.warning("No market data found in PostgreSQL!")
        return pd.DataFrame()
        
    market_df["ngay"] = pd.to_datetime(market_df["ngay"])
    market_df = market_df.rename(columns={
        "ngay": "Ngay",
        "open": "index_open",
        "high": "index_high",
        "low": "index_low",
        "close": "index_close"
    })
    
    # Build market features and shift by 1 day to prevent lookahead bias
    df_market_features = build_market_features(market_df, shift=True)

    df_market_features = (
        df_market_features
        .sort_values("Ngay")
        .tail(FEATURE_WINDOW)
        .reset_index(drop=True)
    )
    
    # Save to Redis
    save_market_features(redis_client, df_market_features)
    logger.info(f"Saved market features up to {df_market_features['Ngay'].max().date()} to Redis.")
    return df_market_features

def compute_stock_features_df(df_stock: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """Run the entire stock feature engineering pipeline for a single stock."""
    if df_stock.empty:
        return df_stock

    df = df_stock.copy()
    df["ngay"] = pd.to_datetime(df["ngay"])
    df["Ngay"] = df["ngay"] # Provide both versions for compatibility
    
    # Merge with market features
    # market features are already shifted 1 day in build_market_features
    regime_cols = [c for c in df_market.columns if c not in ["Ngay", "ngay"]]
    df = df.merge(df_market[["Ngay"] + regime_cols], on="Ngay", how="left")
    
    # 1. BCD Features & Event Detection
    df = BCDFeatureEngine.build(df, lookback_peak=60, ma_ma=200)
    df = BCDEventEngine.build(df, lookback=20, drop_pct=0.15, confirm_days=2, max_bc_days=20, max_cd_days=20)
    
    # 2. Breakout Features & Event Detection
    df = BreakoutEventEngine.build(
        df,
        breakout_lookback=100,
        min_base_length=12,
        max_base_length=120,
        cooldown_days=12
    )
    df = build_features_sepa(df)
    
    return df

def update_all_stock_features(engine, redis_client, force_recompute=False):
    """
    Check PostgreSQL and Redis latest dates for all stocks.
    Perform incremental updates if 1-5 days of new data is found.
    Perform full recomputations if new stock, forced, or data mismatch is detected.
    """
    # Load market features first
    df_market = load_market_features(redis_client)
    if df_market.empty:
        df_market = update_market_features_store(engine, redis_client)
        
    # Get all stock_ids from db
    with engine.connect() as conn:
        r = conn.execute(text("SELECT DISTINCT stock_id FROM ohlcv"))
        stock_ids = [row[0] for row in r]
        
    logger.info(f"Found {len(stock_ids)} stocks in database to update.")
    
    for stock_id in stock_ids:
        try:
            # Check latest date in DB
            with engine.connect() as conn:
                r_date = conn.execute(
                    text("SELECT MAX(ngay) FROM ohlcv WHERE stock_id = :stock_id"),
                    {"stock_id": stock_id}
                ).scalar()
            
            if r_date is None:
                continue
                
            db_latest_date = pd.Timestamp(r_date)
            
            # Load from Redis
            df_redis = load_stock_features(redis_client, stock_id)
            
            do_full = force_recompute or df_redis.empty
            do_incremental = False
            
            if not df_redis.empty:
                redis_latest_date = pd.Timestamp(df_redis["ngay"].max())
                
                if redis_latest_date == db_latest_date:
                    logger.info(f"Stock {stock_id} is up to date ({db_latest_date.date()}).")
                    continue
                elif redis_latest_date < db_latest_date:
                    days_diff = (db_latest_date - redis_latest_date).days
                    # If diff is small, do incremental update
                    if days_diff <= 5:
                        do_incremental = True
                    else:
                        do_full = True
                else:
                    # Redis is newer than DB (unexpected, do full reload)
                    do_full = True
                    
            if do_full:
                logger.info(f"Recomputing full history for {stock_id}...")
                query = """
                    SELECT ngay, stock_id, open, high, low, close, volume
                    FROM ohlcv
                    WHERE stock_id = :stock_id
                    ORDER BY ngay
                """
                with engine.connect() as conn:
                    df_db = pd.read_sql(text(query), conn, params={"stock_id": stock_id})
                
                df_features = compute_stock_features_df(df_db, df_market)
                df_features = (
                    df_features
                    .sort_values("ngay")
                    .tail(FEATURE_WINDOW)
                    .reset_index(drop=True)
                )
                save_stock_features(redis_client, stock_id, df_features)
                logger.info(f"Successfully recomputed {len(df_features)} rows for {stock_id}.")
                
            elif do_incremental:
                logger.info(f"Incrementally updating {stock_id}...")
                # Load latest 300 rows from Redis
                df_history = df_redis.tail(FEATURE_WINDOW).copy()
                
                # Load new rows from DB
                query = """
                    SELECT ngay, stock_id, open, high, low, close, volume
                    FROM ohlcv
                    WHERE stock_id = :stock_id AND ngay > :last_date
                    ORDER BY ngay
                """
                with engine.connect() as conn:
                    df_new = pd.read_sql(
                        text(query), conn, 
                        params={"stock_id": stock_id, "last_date": df_redis["ngay"].max().date()}
                    )
                
                if df_new.empty:
                    continue
                    
                # Concatenate history + new rows
                # We only need enough historical columns (prices) to compute the new features
                df_combined = pd.concat([df_history[["ngay", "stock_id", "open", "high", "low", "close", "volume"]], df_new], ignore_index=True)
                
                # Compute features on the combined block
                df_computed = compute_stock_features_df(df_combined, df_market)
                
                # Slice out only the new rows
                df_new_features = df_computed.tail(len(df_new))
                
                # Append to original Redis data
                df_updated = pd.concat([df_redis, df_new_features], ignore_index=True)
                df_updated = (
                    df_updated
                    .sort_values("ngay")
                    .tail(FEATURE_WINDOW)
                    .reset_index(drop=True)
                )
                save_stock_features(redis_client, stock_id, df_updated)
                logger.info(f"Successfully appended {len(df_new)} rows to {stock_id}.")
                
        except Exception as e:
            logger.error(f"Error updating stock {stock_id}: {e}", exc_info=True)
