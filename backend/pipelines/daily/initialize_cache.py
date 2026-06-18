import logging
import os
import sys
import warnings

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


from backend.connectors.database_connection import get_engine
from backend.connectors.redis_client import get_redis_client
from backend.pipelines.processors.feature_updater import update_market_features_store, update_all_stock_features

logger = logging.getLogger("initialize_cache")

def main():
    logger.info("Connecting to database and Redis...")
    engine = get_engine()
    redis_client = get_redis_client()
    
    logger.info("Step 1: Building and caching market regime features (VNINDEX) in Redis...")
    update_market_features_store(engine, redis_client)
    
    logger.info("Step 2: Building and caching stock features for all stocks (full history) in Redis...")
    # Using force_recompute=True to force full computation and overwrite any existing keys
    update_all_stock_features(engine, redis_client, force_recompute=True)
    
    logger.info("Redis Feature Store cache successfully initialized!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
