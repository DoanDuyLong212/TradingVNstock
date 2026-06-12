from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL
)

def get_engine():
    return engine
