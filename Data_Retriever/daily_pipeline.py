import logging
import time
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from vnstock import Market, Quote

from database_connection import get_engine

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
TABLE_NAME = "ohlcv"
DATE_COL = "ngay"
ID_COL = "stock_id"
COMPARE_COLS = ["open", "high", "low", "close", "volume"]

VN_TZ = ZoneInfo("Asia/Bangkok")


# =========================
# HELPERS
# =========================
def today_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz=VN_TZ).normalize()


def to_date_only(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def normalize_api_df(df: pd.DataFrame, stock_id: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "time" in df.columns:
        df = df.rename(columns={"time": DATE_COL})
    elif "date" in df.columns:
        df = df.rename(columns={"date": DATE_COL})
    elif DATE_COL not in df.columns:
        raise ValueError("Không tìm thấy cột time/date trong dataframe API")

    df[DATE_COL] = to_date_only(df[DATE_COL])

    if stock_id is not None:
        df[ID_COL] = stock_id

    df = df.dropna(subset=[DATE_COL]).reset_index(drop=True)

    cols = list(df.columns)
    if ID_COL in cols:
        cols = [ID_COL] + [c for c in cols if c != ID_COL]
    if DATE_COL in cols:
        cols = [DATE_COL] + [c for c in cols if c != DATE_COL]

    return df[cols].reset_index(drop=True)


def load_stock_list(path: str = "data/stock_list.csv") -> list[str]:
    df = pd.read_csv(path)
    if "stock_id" not in df.columns:
        raise ValueError("File stock_list.csv phải có cột stock_id")

    stock_list = (
        df["stock_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    return list(dict.fromkeys(stock_list))


def get_market_latest_2_days(mkt: Market) -> pd.DataFrame:
    df_vnindex = mkt.index("VNINDEX").ohlcv(length=2, interval="1D")
    df_vnindex = normalize_api_df(df_vnindex)

    if df_vnindex.empty or len(df_vnindex) < 2:
        raise ValueError("Không lấy được đủ 2 ngày dữ liệu VNINDEX")

    return df_vnindex.sort_values(DATE_COL).reset_index(drop=True)


def market_is_open_today(df_vnindex_2days: pd.DataFrame) -> tuple[bool, pd.Timestamp, pd.Timestamp]:
    latest_market_date = pd.Timestamp(df_vnindex_2days.iloc[-1][DATE_COL])
    prev_market_date = pd.Timestamp(df_vnindex_2days.iloc[-2][DATE_COL])

    is_trading_today = latest_market_date.date() == today_local().date()
    if latest_market_date.date() < today_local().date():
        logger.warning("Market data lagging - possible holiday or delay")

    return is_trading_today, latest_market_date, prev_market_date


def get_db_df_by_date(engine, target_date: pd.Timestamp) -> pd.DataFrame:
    sql = text(f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {DATE_COL} = :target_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"target_date": target_date.date()})


def delete_rows_by_stock_ids(engine, stock_ids: list[str]):
    if not stock_ids:
        return

    placeholders = ", ".join([f":s{i}" for i in range(len(stock_ids))])
    sql = text(f"""
        DELETE FROM {TABLE_NAME}
        WHERE {ID_COL} IN ({placeholders})
    """)
    params = {f"s{i}": sid for i, sid in enumerate(stock_ids)}

    with engine.begin() as conn:
        conn.execute(sql, params)


def delete_rows_by_stock_ids_and_date(engine, stock_ids: list[str], target_date: pd.Timestamp):
    if not stock_ids:
        return

    placeholders = ", ".join([f":s{i}" for i in range(len(stock_ids))])
    sql = text(f"""
        DELETE FROM {TABLE_NAME}
        WHERE {ID_COL} IN ({placeholders})
          AND {DATE_COL} = :target_date
    """)
    params = {f"s{i}": sid for i, sid in enumerate(stock_ids)}
    params["target_date"] = target_date.date()

    with engine.begin() as conn:
        conn.execute(sql, params)


def insert_df(engine, df: pd.DataFrame):
    if df is None or df.empty:
        return

    df_to_save = df.copy()
    if DATE_COL in df_to_save.columns:
        df_to_save[DATE_COL] = pd.to_datetime(df_to_save[DATE_COL], errors="coerce").dt.date

    with engine.begin() as conn:
        df_to_save.to_sql(
            TABLE_NAME,
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )


def safe_fetch_2_latest_stock_rows(
    mkt: Market,
    stock_id: str,
    max_retry: int = 3,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    for attempt in range(1, max_retry + 1):
        try:
            df = mkt.equity(stock_id).ohlcv(length=2, interval="1D")
            df = normalize_api_df(df, stock_id=stock_id)

            if df is not None and not df.empty:
                logger.info(stock_id)
                return df

        except Exception:
            pass

        if attempt < max_retry:
            time.sleep(sleep_seconds * attempt)

    logger.warning(f"{stock_id} ---> fail")
    return pd.DataFrame()


def safe_fetch_full_history_stock(
    stock_id: str,
    end_date: pd.Timestamp,
    max_retry: int = 3,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    end_plus_1 = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for attempt in range(1, max_retry + 1):
        try:
            df = Quote(symbol=stock_id, source="KBS").history(
                start="2000-01-01",
                end=end_plus_1,
                interval="1D",
            )
            df = normalize_api_df(df, stock_id=stock_id)

            if df is not None and not df.empty:
                logger.info(stock_id)
                return df

        except Exception:
            pass

        if attempt < max_retry:
            time.sleep(sleep_seconds * attempt)

    logger.warning(f"{stock_id} ---> fail")
    return pd.DataFrame()


def split_compare_today(
    df_2latest: pd.DataFrame,
    market_prev_date: pd.Timestamp,
    market_today: pd.Timestamp,
):
    if df_2latest.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_2latest = df_2latest.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    df_compare = df_2latest[df_2latest[DATE_COL] == market_prev_date.date()].copy()
    df_today = df_2latest[df_2latest[DATE_COL] == market_today.date()].copy()

    return df_compare.reset_index(drop=True), df_today.reset_index(drop=True)


def compare_compare_and_yesterday(
    df_compare: pd.DataFrame,
    df_yesterday: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_compare.empty:
        return pd.DataFrame(), pd.DataFrame(columns=[ID_COL])

    if df_yesterday.empty:
        mismatched = df_compare[[ID_COL]].drop_duplicates().copy()
        mismatched["reason"] = "yesterday_missing_in_db"
        return pd.DataFrame(), mismatched

    compare_map = df_compare.drop_duplicates(subset=[ID_COL]).set_index(ID_COL)
    yesterday_map = df_yesterday.drop_duplicates(subset=[ID_COL]).set_index(ID_COL)

    matched_ids = []
    mismatched_ids = []

    for stock_id, row in compare_map.iterrows():
        if stock_id not in yesterday_map.index:
            mismatched_ids.append(stock_id)
            continue

        db_row = yesterday_map.loc[stock_id]
        ok = True

        for col in COMPARE_COLS:
            if col not in row.index or col not in db_row.index:
                continue

            v1 = pd.to_numeric(row[col], errors="coerce")
            v2 = pd.to_numeric(db_row[col], errors="coerce")

            if pd.isna(v1) and pd.isna(v2):
                continue

            if pd.isna(v1) or pd.isna(v2) or round(float(v1), 6) != round(float(v2), 6):
                ok = False
                break

        if ok:
            matched_ids.append(stock_id)
        else:
            mismatched_ids.append(stock_id)

    matched_df = pd.DataFrame({ID_COL: matched_ids})
    mismatched_df = pd.DataFrame({ID_COL: mismatched_ids})
    mismatched_df["reason"] = "value_mismatch_or_missing"

    return matched_df, mismatched_df


# =========================
# MAIN PIPELINE
# =========================
def run_daily_pipeline():
    engine = get_engine()
    stock_list = load_stock_list("data/stock_list.csv")
    mkt = Market()

    # 1) Lấy 2 ngày mới nhất của thị trường
    df_vnindex_2days = get_market_latest_2_days(mkt)
    is_trading_today, market_today, market_prev_date = market_is_open_today(df_vnindex_2days)

    logger.info(f"Market latest date: {market_today.date()}, previous market date: {market_prev_date.date()}")
    if not is_trading_today:
        logger.info("Hôm nay thị trường không giao dịch -> dừng pipeline.")
        return

    # 2) Lấy dữ liệu 2 ngày mới nhất của từng mã
    all_2latest = []
    MAX_RETRY = 3

    for stock_id in stock_list:
        df_2 = safe_fetch_2_latest_stock_rows(
            mkt=mkt,
            stock_id=stock_id,
            max_retry=MAX_RETRY,
            sleep_seconds=1.0,
        )

        if not df_2.empty:
            all_2latest.append(df_2)

        time.sleep(1)

    if not all_2latest:
        logger.warning("Không lấy được dữ liệu nào từ API.")
        return

    df_2latest = pd.concat(all_2latest, ignore_index=True)
    df_2latest = df_2latest.drop_duplicates(subset=[ID_COL, DATE_COL]).reset_index(drop=True)

    # 3) Kiểm tra đủ 2 ngày cần so sánh
    available_dates = set(pd.to_datetime(df_2latest[DATE_COL], errors="coerce").dt.date.dropna().tolist())
    if market_prev_date.date() not in available_dates:
        logger.warning("Thiếu ngày market_prev_date trong dữ liệu API.")
        return
    if market_today.date() not in available_dates:
        logger.warning("Thiếu ngày market_today trong dữ liệu API.")
        return

    # 4) Tách df_compare và df_today
    df_compare, df_today = split_compare_today(df_2latest, market_prev_date, market_today)

    if df_compare.empty or df_today.empty:
        logger.warning("Không tách được đủ df_compare / df_today theo 2 ngày thị trường.")
        return

    # 5) Lấy dữ liệu yesterday từ DB
    df_yesterday = get_db_df_by_date(engine, market_prev_date)

    # 6) So sánh df_compare với df_yesterday
    matched_df, mismatched_df = compare_compare_and_yesterday(df_compare, df_yesterday)

    matched_ids = matched_df[ID_COL].dropna().astype(str).tolist()
    mismatched_ids = mismatched_df[ID_COL].dropna().astype(str).tolist()

    logger.info(f"Matched: {len(matched_ids)} | Mismatched: {len(mismatched_ids)}")

    # 7) Với mã khớp: chỉ nạp df_today
    if matched_ids:
        df_today_matched = df_today[df_today[ID_COL].astype(str).isin(matched_ids)].copy()
        delete_rows_by_stock_ids_and_date(engine, matched_ids, market_today)
        insert_df(engine, df_today_matched)
        logger.info(f"Inserted today rows for {len(matched_ids)} matched stocks.")

    # 8) Với mã mismatch: tải lại toàn bộ lịch sử và thay thế
    if mismatched_ids:
        full_history_rows = []

        for stock_id in mismatched_ids:
            full_df = safe_fetch_full_history_stock(
                stock_id=stock_id,
                end_date=market_today,
                max_retry=3,
                sleep_seconds=1.0,
            )

            if not full_df.empty:
                full_history_rows.append(full_df)

            time.sleep(1)

        if full_history_rows:
            delete_rows_by_stock_ids(engine, mismatched_ids)

            full_history_df = pd.concat(full_history_rows, ignore_index=True)
            full_history_df = full_history_df.drop_duplicates(subset=[ID_COL, DATE_COL]).reset_index(drop=True)

            insert_df(engine, full_history_df)
            logger.info(f"Reloaded full history for {len(mismatched_ids)} mismatch stocks.")

    logger.info("Pipeline hoàn tất.")


if __name__ == "__main__":
    run_daily_pipeline()