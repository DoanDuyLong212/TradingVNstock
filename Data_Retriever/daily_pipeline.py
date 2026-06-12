import time
import pandas as pd
from datetime import date, timedelta
from vnstock.api.quote import Quote
from database_connection import get_connection


# =========================
# CONFIG
# =========================

TODAY_CAL = date.today()
TODAY_STR = TODAY_CAL.strftime("%Y-%m-%d")

STOCK_PATH = "Data_Retriever/data/stock_list.csv"

REQUEST_SLEEP = 0.7
RETRY_FETCH = 2
MARKET_LOOKBACK_DAYS = 12
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
# MARKET STATUS
# =========================

def get_market_end_date():
    start = (TODAY_CAL - timedelta(days=MARKET_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    try:
        q = Quote(symbol="VNINDEX", source="KBS")
        df = q.history(start=start, end=TODAY_STR, interval="1D").copy()

        if df is None or df.empty:
            return TODAY_CAL - timedelta(days=1), False

        df["Ngay"] = pd.to_datetime(df["time"]).dt.date
        df = df.drop(columns=["time"])

        market_end_date = df["Ngay"].max()
        market_open_today = (market_end_date == TODAY_CAL)

        print(f"📈 Market latest date: {market_end_date} | Open today: {market_open_today}")

        return market_end_date, market_open_today

    except Exception as e:
        print(f"⚠️ Market error: {e}")
        return TODAY_CAL - timedelta(days=1), False


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

    conn = get_connection()

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

    df = pd.read_sql(query, conn)

    conn.close()

    if len(df):
        df["Ngay"] = pd.to_datetime(df["Ngay"]).dt.date

    return clean(df)



def build_df_yesterday(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["stock_id","Ngay","open","high","low","close","volume"])

    return (
        df.sort_values(["stock_id","Ngay"])
          .drop_duplicates(["stock_id"], keep="last")
          .reset_index(drop=True)
    )



# =========================
# FETCH
# =========================

def fetch(symbol, start, end):
    for i in range(RETRY_FETCH):
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
            return None


def fetch_check(symbol, market_end_date):
    start = (market_end_date - timedelta(days=CHECK_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = market_end_date.strftime("%Y-%m-%d")

    df = fetch(symbol, start, end)

    if df is None or df.empty:
        return None

    return df.drop_duplicates(["stock_id","Ngay"]).sort_values("Ngay").tail(2).reset_index(drop=True)


# =========================
# BUILD PAYLOAD
# =========================

def build_symbol_payload(symbol, local_full, local_yesterday_row, check_df, market_end_date):

    full_start = "2000-01-01"
    full_end = market_end_date.strftime("%Y-%m-%d")

    # ❌ FIRST LOAD -> SKIP (không FULL nữa)
    if local_full.empty:

        full_df = fetch(
            symbol,
            "2000-01-01",
            market_end_date.strftime("%Y-%m-%d")
        )

        return {
            "symbol": symbol,
            "mode": "FIRST_LOAD",
            "db_df": full_df,
            "metadata_df": full_df,
            "check_df": check_df
        }

    if check_df is None or len(check_df) < 2:
        return {
            "symbol": symbol,
            "mode": "SKIP",
            "db_df": None,
            "metadata_df": local_full,
            "check_df": check_df
        }
    
    if local_yesterday_row.empty:
        full_df = fetch(
            symbol,
            "2000-01-01",
            market_end_date.strftime("%Y-%m-%d")
        )

        return {
            "symbol": symbol,
            "mode": "FIRST_LOAD",
            "db_df": full_df,
            "metadata_df": full_df,
            "check_df": check_df
        }

    api_yesterday = check_df.iloc[-2]
    local_yesterday = local_yesterday_row.iloc[0]

    # =========================
    # ONLY FULL CONDITION
    # =========================
    if not rows_match(local_yesterday, api_yesterday):

        full_df = fetch(symbol, full_start, full_end)

        if full_df is None or full_df.empty:
            return {
                "symbol": symbol,
                "mode": "SKIP",
                "db_df": None,
                "metadata_df": local_full,
                "check_df": check_df
            }

        return {
            "symbol": symbol,
            "mode": "FULL",
            "db_df": full_df,
            "metadata_df": full_df,
            "check_df": check_df
        }

    # =========================
    # INCR LOGIC
    # =========================
    last_date = local_yesterday["Ngay"]

    new_rows = check_df[check_df["Ngay"] > last_date].copy()
    new_rows = clean(new_rows)

    if new_rows.empty:
        return {
            "symbol": symbol,
            "mode": "NOOP",
            "db_df": None,
            "metadata_df": local_full,
            "check_df": check_df
        }

    updated = pd.concat([local_full, new_rows], ignore_index=True)
    updated = clean(updated)

    return {
        "symbol": symbol,
        "mode": "INCR",
        "db_df": new_rows,
        "metadata_df": updated,
        "check_df": check_df
    }







def upsert_ohlcv(df: pd.DataFrame):
    if df is None or df.empty:
        return

    conn = get_connection()
    cur = conn.cursor()

    query = """
    INSERT INTO ohlcv (stock_id, ngay, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (stock_id, ngay)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;
    """

    data = df[[
        "stock_id", "Ngay", "open", "high", "low", "close", "volume"
    ]].values.tolist()

    cur.executemany(query, data)

    conn.commit()
    cur.close()
    conn.close()


# =========================
# RUN
# =========================

def run():

    print("🚀 START PIPELINE")

    metadata = load_metadata_from_db()

    market_end_date, _ = get_market_end_date()

    df_yesterday = build_df_yesterday(metadata)

    for i, symbol in enumerate(symbols):

        local_full = (
            metadata[metadata["stock_id"] == symbol]
            .copy()
            .sort_values("Ngay")
        )

        local_yesterday_row = (
            df_yesterday[df_yesterday["stock_id"] == symbol]
            .copy()
        )

        check_df = fetch_check(symbol, market_end_date)

        payload = build_symbol_payload(
            symbol,
            local_full,
            local_yesterday_row,
            check_df,
            market_end_date
        )

        if payload["db_df"] is not None and not payload["db_df"].empty:
            upsert_ohlcv(payload["db_df"])

        print(f"[{i}] {symbol} -> {payload['mode']}")

    print("DONE")


if __name__ == "__main__":
    run()