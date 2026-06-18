import os
from dotenv import load_dotenv
from upstash_redis import Redis

load_dotenv()

_redis_client = None

def get_redis_client() -> Redis:
    """Initialize and return the Upstash Redis client using .env configuration."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    
    if not url or not token:
        raise ValueError("UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN must be set in .env")
        
    _redis_client = Redis(url=url, token=token)
    return _redis_client
