from dotenv import load_dotenv
import os
import psycopg
import time

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    print("⏳ connecting DB...")

    start = time.time()
    conn = psycopg.connect(DATABASE_URL)
    print("✅ connected in", time.time() - start, "s")

    return conn
