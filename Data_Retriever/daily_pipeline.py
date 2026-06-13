import os
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from database_connection import get_engine

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
from vnstock import Market, Quote

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
TABLE_NAME = "ohlcv"   # đổi thành tên bảng thật của bạn
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
    """
    - đổi cột time/date -> ngay
    - bỏ time component
    - giữ date dạng Python date
    - thêm stock_id nếu cần
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # chuẩn hóa tên cột ngày
    if "time" in df.columns:
        df = df.rename(columns={"time": DATE_COL})
    elif "date" in df.columns:
        df = df.rename(columns={"date": DATE_COL})
    elif DATE_COL not in df.columns:
        raise ValueError("Không tìm thấy cột time/date trong dataframe API")

    df[DATE_COL] = to_date_only(df[DATE_COL])

    if stock_id is not None:
        df[ID_COL] = stock_id

    # bỏ các dòng lỗi ngày
    df = df.dropna(subset=[DATE_COL])

    # đưa stock_id lên đầu cho dễ nhìn
    cols = list(df.columns)
    if ID_COL in cols:
        cols = [ID_COL] + [c for c in cols if c != ID_COL]
    if DATE_COL in cols:
        cols = [DATE_COL] + [c for c in cols if c != DATE_COL]
    df = df[cols]

    return df.reset_index(drop=True)


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
    # loại trùng
    return list(dict.fromkeys(stock_list))


def get_market_latest_2_days() -> pd.DataFrame:
    """
    Lấy 2 ngày mới nhất của VNINDEX để xác định:
    - hôm nay có giao dịch hay không
    - ngày giao dịch trước của thị trường là ngày nào
    """
    mkt = Market()
    df_vnindex = mkt.index("VNINDEX").ohlcv(length=2, interval="1D")
    df_vnindex = normalize_api_df(df_vnindex)

    if df_vnindex.empty or len(df_vnindex) < 2:
        raise ValueError("Không lấy được đủ 2 ngày dữ liệu VNINDEX")

    df_vnindex = df_vnindex.sort_values(DATE_COL).reset_index(drop=True)
    return df_vnindex


def market_is_open_today(df_vnindex_2days: pd.DataFrame) -> tuple[bool, pd.Timestamp, pd.Timestamp]:
    """
    Nếu ngày mới nhất từ API = ngày hôm nay -> có giao dịch hôm nay
    ngược lại -> hôm nay không giao dịch
    """
    latest_market_date = pd.Timestamp(df_vnindex_2days.iloc[-1][DATE_COL])
    prev_market_date = pd.Timestamp(df_vnindex_2days.iloc[-2][DATE_COL])
    today = today_local()

    is_trading_today = latest_market_date == today
    return is_trading_today, latest_market_date, prev_market_date


def get_db_latest_dates(engine) -> pd.DataFrame:
    """
    Lấy ngày mới nhất trong DB cho từng stock.
    """
    sql = text(f"""
        SELECT {ID_COL}, MAX({DATE_COL}) AS latest_date
        FROM {TABLE_NAME}
        GROUP BY {ID_COL}
        ORDER BY {ID_COL}
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if not df.empty:
        df[DATE_COL] = pd.to_datetime(df["latest_date"], errors="coerce").dt.date
        df = df.drop(columns=["latest_date"])
    return df


def get_db_df_by_date(engine, target_date: pd.Timestamp) -> pd.DataFrame:
    sql = text(f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {DATE_COL} = :target_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"target_date": target_date.date()})
    return df


def get_db_df_by_stocks(engine, stock_ids: list[str]) -> pd.DataFrame:
    if not stock_ids:
        return pd.DataFrame()

    placeholders = ", ".join([f":s{i}" for i in range(len(stock_ids))])
    sql = text(f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE {ID_COL} IN ({placeholders})
    """)
    params = {f"s{i}": sid for i, sid in enumerate(stock_ids)}
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df


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


def fetch_2_latest_stock_rows(stock_id: str) -> pd.DataFrame:
    """
    Lấy 2 giá trị mới nhất của 1 cổ phiếu.
    """
    mkt = Market()
    df = mkt.equity(stock_id).ohlcv(length=2, interval="1D")
    return normalize_api_df(df, stock_id=stock_id)


def fetch_full_history_stock(stock_id: str, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Tải toàn bộ lịch sử của 1 mã từ 2000-01-01 đến end_date + 1.
    """
    end_plus_1 = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = Quote(symbol=stock_id, source="KBS").history(
        start="2000-01-01",
        end=end_plus_1,
        interval="1D",
    )
    df = normalize_api_df(df, stock_id=stock_id)
    return df


def split_compare_today(df_2latest: pd.DataFrame, market_prev_date: pd.Timestamp, market_today: pd.Timestamp):
    """
    Tách dataframe 2 ngày thành:
    - df_compare: ngày nhỏ hơn (yesterday market day)
    - df_today: ngày lớn hơn (today market day)
    """
    if df_2latest.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_2latest = df_2latest.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    df_compare = df_2latest[df_2latest[DATE_COL] == market_prev_date.date()].copy()
    df_today = df_2latest[df_2latest[DATE_COL] == market_today.date()].copy()

    return df_compare.reset_index(drop=True), df_today.reset_index(drop=True)


def compare_compare_and_yesterday(df_compare: pd.DataFrame, df_yesterday: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    So sánh df_compare với df_yesterday theo stock_id và các cột numeric chính.
    Trả về:
    - matched_rows
    - mismatched_stock_ids_df
    """
    if df_compare.empty:
        return pd.DataFrame(), pd.DataFrame(columns=[ID_COL])

    if df_yesterday.empty:
        mismatched = df_compare[[ID_COL]].drop_duplicates().copy()
        mismatched["reason"] = "yesterday_missing_in_db"
        return pd.DataFrame(), mismatched

    # Chỉ giữ các cột có thật trong cả 2 dataframe
    cols_to_compare = [c for c in COMPARE_COLS if c in df_compare.columns and c in df_yesterday.columns]

    merged = df_compare.merge(
        df_yesterday[[ID_COL, DATE_COL] + cols_to_compare].drop_duplicates(subset=[ID_COL]),
        on=ID_COL,
        how="left",
        suffixes=("_cmp", "_db"),
    )

    mismatch_mask = merged[f"{DATE_COL}_db"].isna()

    for col in cols_to_compare:
        a = merged[f"{col}_cmp"]
        b = merged[f"{col}_db"]

        # so sánh an toàn cho số thực
        a_norm = pd.to_numeric(a, errors="coerce").round(6)
        b_norm = pd.to_numeric(b, errors="coerce").round(6)

        mismatch_mask |= ~(a_norm.fillna(-999999) == b_norm.fillna(-999999))

    matched = merged.loc[~mismatch_mask, [ID_COL]].copy()
    mismatched = merged.loc[mismatch_mask, [ID_COL]].drop_duplicates().copy()
    mismatched["reason"] = "value_mismatch_or_missing"

    return matched, mismatched


# =========================
# MAIN PIPELINE
# =========================
def run_daily_pipeline():
    engine = get_engine()
    stock_list = load_stock_list("data/stock_list.csv")

    # 1) Lấy 2 ngày mới nhất của thị trường
    df_vnindex_2days = get_market_latest_2_days()
    is_trading_today, market_today, market_prev_date = market_is_open_today(df_vnindex_2days)

    logger.info(f"Market latest date: {market_today.date()}, previous market date: {market_prev_date.date()}")
    if not is_trading_today:
        logger.info("Hôm nay thị trường không giao dịch -> dừng pipeline.")
        return

    # 2) Lấy dữ liệu 2 ngày mới nhất của từng mã
    all_2latest = []
    for stock_id in stock_list:
        try:
            df_2 = fetch_2_latest_stock_rows(stock_id)
            if not df_2.empty:
                all_2latest.append(df_2)
        except Exception as e:
            logger.warning(f"Lỗi lấy 2 ngày mới nhất của {stock_id}: {e}")

    if not all_2latest:
        logger.warning("Không lấy được dữ liệu nào từ API.")
        return

    df_2latest = pd.concat(all_2latest, ignore_index=True)
    df_2latest = df_2latest.drop_duplicates(subset=[ID_COL, DATE_COL]).reset_index(drop=True)

    # 3) Tách ra df_compare và df_today
    df_compare, df_today = split_compare_today(df_2latest, market_prev_date, market_today)

    if df_compare.empty or df_today.empty:
        logger.warning("Không tách được đủ df_compare / df_today theo 2 ngày thị trường.")
        return

    # 4) Lấy dữ liệu yesterday từ DB
    #    Nếu DB đang có ngày market_prev_date thì dùng nó để so sánh
    df_yesterday = get_db_df_by_date(engine, market_prev_date)

    # 5) So sánh df_compare với df_yesterday
    matched_df, mismatched_df = compare_compare_and_yesterday(df_compare, df_yesterday)

    matched_ids = matched_df[ID_COL].dropna().astype(str).tolist()
    mismatched_ids = mismatched_df[ID_COL].dropna().astype(str).tolist()

    logger.info(f"Matched: {len(matched_ids)} | Mismatched: {len(mismatched_ids)}")

    # 6) Với mã khớp: chỉ nạp df_today
    if matched_ids:
        df_today_matched = df_today[df_today[ID_COL].astype(str).isin(matched_ids)].copy()

        # tránh trùng ngày hôm nay
        delete_rows_by_stock_ids_and_date(engine, matched_ids, market_today)
        insert_df(engine, df_today_matched)

        logger.info(f"Đã insert df_today cho {len(matched_ids)} mã khớp.")

    # 7) Với mã mismatch: tải lại toàn bộ lịch sử và thay thế
    if mismatched_ids:
        full_history_rows = []
        for stock_id in mismatched_ids:
            try:
                full_df = fetch_full_history_stock(stock_id, market_today)
                if not full_df.empty:
                    full_history_rows.append(full_df)
            except Exception as e:
                logger.warning(f"Lỗi tải full history của {stock_id}: {e}")

        if full_history_rows:
            # xóa toàn bộ dữ liệu cũ của các mã mismatch rồi nạp lại
            delete_rows_by_stock_ids(engine, mismatched_ids)

            full_history_df = pd.concat(full_history_rows, ignore_index=True)
            full_history_df = full_history_df.drop_duplicates(subset=[ID_COL, DATE_COL]).reset_index(drop=True)

            insert_df(engine, full_history_df)
            logger.info(f"Đã reload full history cho {len(mismatched_ids)} mã mismatch.")

    logger.info("Pipeline hoàn tất.")


if __name__ == "__main__":
    run_daily_pipeline()
