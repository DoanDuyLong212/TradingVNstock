import os
import time
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()

logger = logging.getLogger(__name__)

_engine = None

def get_engine(retries: int = 3, delay: float = 2.0):
    global _engine
    if _engine is not None:
        return _engine

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")

    for attempt in range(retries):
        try:
            _engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_timeout=30,
                connect_args={
                    "connect_timeout": 10,
                },
            )

            with _engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"Database connection established (attempt {attempt + 1})")
            return _engine

        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.error(f"Failed to connect to database after {retries} attempts")
                raise

    return _engine
