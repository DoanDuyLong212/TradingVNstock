"""
Daily metadata sync

Sync:
- dim_stock
- dim_icb

Source:
- data/stock_list.csv
"""

import pandas as pd
from vnstock import Reference
from sqlalchemy import text

from Data_Retriever.database_connection import get_engine


def sync_stock_metadata():
    engine = get_engine()
    ref = Reference()

    # ==================================================
    # 1. LOAD UNIVERSE
    # ==================================================
    stock_list = (
        pd.read_csv("Data_Retriever/data/stock_list.csv")["stock_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Loaded {len(stock_list)} stocks")

    # ==================================================
    # 2. BUILD DIM_STOCK
    # ==================================================
    stock_df = ref.equity.list_by_exchange()

    if "symbol" in stock_df.columns:
        stock_df = stock_df.rename(columns={"symbol": "stock_id"})

    if "exchange" not in stock_df.columns:
        if "floor" in stock_df.columns:
            stock_df = stock_df.rename(columns={"floor": "exchange"})
        else:
            raise ValueError("exchange/floor column not found")

    stock_df = (
        stock_df[
            ["stock_id", "exchange"]
        ]
        .drop_duplicates()
    )

    stock_df = stock_df[
        stock_df["stock_id"].isin(stock_list)
    ].reset_index(drop=True)

    # ==================================================
    # 3. INDEX GROUP
    # ==================================================
    vn30 = set(ref.equity.list_by_group("VN30"))
    vn100 = set(ref.equity.list_by_group("VN100"))
    vnall = set(ref.equity.list_by_group("VNALL"))

    stock_df["stock_group"] = "Other"

    stock_df.loc[
        stock_df["stock_id"].isin(vnall),
        "stock_group"
    ] = "VNALL"

    stock_df.loc[
        stock_df["stock_id"].isin(vn100),
        "stock_group"
    ] = "VN100"

    stock_df.loc[
        stock_df["stock_id"].isin(vn30),
        "stock_group"
    ] = "VN30"

    # ==================================================
    # 4. INDUSTRY
    # ==================================================
    industry_df = ref.industry.sectors()

    industry_df = (
        industry_df[
            [
                "symbol",
                "industry_code",
                "industry_name",
            ]
        ]
        .rename(columns={"symbol": "stock_id"})
        .drop_duplicates(subset=["stock_id"])
    )

    stock_df = stock_df.merge(
        industry_df,
        on="stock_id",
        how="left",
    )

    stock_df = stock_df[
        [
            "stock_id",
            "exchange",
            "stock_group",
            "industry_code",
            "industry_name",
        ]
    ]

    # ==================================================
    # 5. UPSERT DIM_STOCK
    # ==================================================
    upsert_stock_sql = """
    INSERT INTO dim_stock (
        stock_id,
        exchange,
        stock_group,
        industry_code,
        industry_name
    )
    VALUES (
        :stock_id,
        :exchange,
        :stock_group,
        :industry_code,
        :industry_name
    )
    ON CONFLICT (stock_id)
    DO UPDATE SET
        exchange      = EXCLUDED.exchange,
        stock_group   = EXCLUDED.stock_group,
        industry_code = EXCLUDED.industry_code,
        industry_name = EXCLUDED.industry_name,
        updated_at    = CURRENT_TIMESTAMP;
    """

    records = stock_df.to_dict("records")

    with engine.begin() as conn:
        conn.execute(text(upsert_stock_sql), records)

    print(f"Synced dim_stock: {len(records)} rows")

    # ==================================================
    # 6. BUILD DIM_ICB
    # ==================================================
    icb_df = ref.equity.list_by_industry()

    icb_df = (
        icb_df[
            [
                "symbol",
                "icb_level",
                "icb_code",
                "icb_name",
            ]
        ]
        .rename(columns={"symbol": "stock_id"})
    )

    icb_df = icb_df[
        icb_df["stock_id"].isin(stock_list)
    ].copy()

    # ==================================================
    # 7. UPSERT DIM_ICB
    # ==================================================
    upsert_icb_sql = """
    INSERT INTO dim_icb (
        stock_id,
        icb_level,
        icb_code,
        icb_name
    )
    VALUES (
        :stock_id,
        :icb_level,
        :icb_code,
        :icb_name
    )
    ON CONFLICT (stock_id, icb_level)
    DO UPDATE SET
        icb_code = EXCLUDED.icb_code,
        icb_name = EXCLUDED.icb_name;
    """

    records = icb_df.to_dict("records")

    with engine.begin() as conn:
        conn.execute(text(upsert_icb_sql), records)

    print(f"Synced dim_icb: {len(records)} rows")

    print("Metadata sync completed")


if __name__ == "__main__":
    sync_stock_metadata()