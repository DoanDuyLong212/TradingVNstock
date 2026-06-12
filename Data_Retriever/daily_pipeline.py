import time
import pandas as pd
from datetime import date, timedelta
from vnstock.api.quote import Quote
from database_connection import get_engine
from sqlalchemy import text


# =========================
# CONFIG
# =========================

CUT_OFF_DATE = date.today() - timedelta(days=1)
CUT_OFF_STR = CUT_OFF_DATE.strftime("%Y-%m-%d")

STOCK_PATH = "Data_Retriever/data/stock_list.csv"

REQUEST_SLEEP = 0.7
RETRY_FETCH = 2
CHECK_LOOKBACK_DAYS = 12


# =========================
# HELPERS
# =========================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[[
        "stock_id", "Ngay",
        "open", "high", "low", "close", "volume"
    ]].copy()

    df = df.drop_duplicates(["stock_id", "Ngay"], keep="last")
    df = df.sort_values(["stock_id", "Ngay"]).reset_index(drop=True)

    return df


def rows_match(local_row: pd.Series, api_row: pd.Series) -> bool:
    if pd.isna(local_row["Ngay"]) or pd.isna(api_row["Ngay"]):
        return False

    if local_row["Ngay"] != api_row["Ngay"]:
        return False

    for col in ["open", "high", "low", "close"]:
        a = local_row[col]
        b = api_row[col]

        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b):
            return False

        try:
            if abs(float(a) - float(b)) > 1e-6:
                return False
        except Exception:
            return False

    a_vol = local_row["volume"]
    b_vol = api_row["volume"]

    if pd.isna(a_vol) and pd.isna(b_vol):
        return True
    if pd.isna(a_vol) != pd.isna(b_vol):
        return False

    try:
        return int(a_vol) == int(b_vol)
    except Exception:
        return False


# =========================
# LOAD DATA
# =========================

symbols = (
    pd.read_csv(STOCK_PATH)["stock_id"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)


def load_metadata_from_db():
    engine = get_engine()

    query = """
    SELECT
        stock_id,
        ngay AS "Ngay",
        open,
        high,
        low,
        close,
        volume
    FROM ohlcv
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        df["Ngay"] = pd.to_datetime(df["Ngay"]).dt.date

    return clean(df)


def build_df_yesterday(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["stock_id", "Ngay", "open", "high", "low", "close", "volume"])

    return (
        df.sort_values(["stock_id", "Ngay"])
          .drop_duplicates(["stock_id"], keep="last")
          .reset_index(drop=True)
    )


# =========================
# FETCH
# =========================

def fetch(symbol, start, end):
    for _ in range(RETRY_FETCH):
        try:
            time.sleep(REQUEST_SLEEP)

            df = Quote(symbol=symbol, source="KBS").history(
                start=start, end=end, interval="1D"
            ).copy()

            if df is None or df.empty:
                return None

            df["Ngay"] = pd.to_datetime(df["time"]).dt.date
            df = df.drop(columns=["time"])
            df["stock_id"] = symbol

            return clean(df)

        except Exception as e:
            print(f"⚠️ fetch error {symbol}: {e}")
            continue

    return None


def fetch_check(symbol):
    start = (CUT_OFF_DATE - timedelta(days=CHECK_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = CUT_OFF_STR

    df = fetch(symbol, start, end)

    if df is None or df.empty:
        return None

    return (
        df.drop_duplicates(["stock_id", "Ngay"])
          .sort_values("Ngay")
          .reset_index(drop=True)
    )


# =========================
# BUILD PAYLOAD
# =========================

def build_symbol_payload(symbol, local_full, check_df):
    full_start = "2000-01-01"
    full_end = CUT_OFF_STR

    if local_full.empty:
        full_df = fetch(symbol, full_start, full_end)

        return {
            "symbol": symbol,
            "mode": "FIRST_LOAD",
            "db_df": full_df,
        }

    if check_df is None or check_df.empty:
        return {
            "symbol": symbol,
            "mode": "SKIP",
            "db_df": None,
        }

    api_cutoff_row = check_df[check_df["Ngay"] == CUT_OFF_DATE].copy()
    if api_cutoff_row.empty:
        return {
            "symbol": symbol,
            "mode": "SKIP",
            "db_df": None,
        }

    local_cutoff_row = local_full[local_full["Ngay"] == CUT_OFF_DATE].copy()

    if not local_cutoff_row.empty:
        if rows_match(local_cutoff_row.iloc[-1], api_cutoff_row.iloc[-1]):
            return {
                "symbol": symbol,
                "mode": "NOOP",
                "db_df": None,
            }

        full_df = fetch(symbol, full_start, full_end)
        if full_df is None or full_df.empty:
            return {
                "symbol": symbol,
                "mode": "SKIP",
                "db_df": None,
            }

        return {
            "symbol": symbol,
            "mode": "FULL",
            "db_df": full_df,
        }

    local_latest_date = local_full["Ngay"].max()

    if local_latest_date < CUT_OFF_DATE:
        inc_start = (local_latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        incr_df = fetch(symbol, inc_start, full_end)

        if incr_df is None or incr_df.empty:
            return {
                "symbol": symbol,
                "mode": "SKIP",
                "db_df": None,
            }

        incr_df = incr_df[incr_df["Ngay"] > local_latest_date].copy()
        incr_df = clean(incr_df)

        if incr_df.empty:
            return {
                "symbol": symbol,
                "mode": "NOOP",
                "db_df": None,
            }

        return {
            "symbol": symbol,
            "mode": "INCR",
            "db_df": incr_df,
        }

    return {
        "symbol": symbol,
        "mode": "NOOP",
        "db_df": None,
    }


# =========================
# UPSERT
# =========================

def upsert_ohlcv(df: pd.DataFrame):
    if df is None or df.empty:
        return

    engine = get_engine()

    df = df.copy()
    df = df.rename(columns={"Ngay": "ngay"})
    df["ngay"] = pd.to_datetime(df["ngay"]).dt.date

    query = text("""
    INSERT INTO ohlcv (stock_id, ngay, open, high, low, close, volume)
    VALUES (:stock_id, :ngay, :open, :high, :low, :close, :volume)
    ON CONFLICT (stock_id, ngay)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
    """)

    records = df[[
        "stock_id", "ngay", "open", "high", "low", "close", "volume"
    ]].to_dict("records")

    with engine.begin() as conn:
        conn.execute(query, records)


# =========================
# RUN
# =========================

def run():
    print("🚀 START PIPELINE")
    print(f"📌 CUT_OFF_DATE = {CUT_OFF_DATE}")

    metadata = load_metadata_from_db()
    df_yesterday = build_df_yesterday(metadata)

    for i, symbol in enumerate(symbols):
        local_full = (
            metadata[metadata["stock_id"] == symbol]
            .copy()
            .sort_values("Ngay")
        )

        check_df = fetch_check(symbol)

        payload = build_symbol_payload(
            symbol,
            local_full,
            check_df
        )

        if payload["db_df"] is not None and not payload["db_df"].empty:
            upsert_ohlcv(payload["db_df"])

        print(f"[{i}] {symbol} -> {payload['mode']}")

    print("DONE")


if __name__ == "__main__":
    run()
