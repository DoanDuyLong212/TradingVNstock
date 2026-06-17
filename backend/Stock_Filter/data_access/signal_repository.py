from sqlalchemy import text


def save_signals(engine, df_signals):
    if df_signals.empty:
        return

    sql = text("""
        INSERT INTO stock_signals
        (
            stock_id,
            signal_date,
            model_name,
            probability
        )
        VALUES
        (
            :stock_id,
            :signal_date,
            :model_name,
            :probability
        )
        ON CONFLICT
        (
            stock_id,
            signal_date,
            model_name
        )
        DO UPDATE SET
            probability = EXCLUDED.probability,
    """)

    records = []

    for _, row in df_signals.iterrows():
        records.append({
            "stock_id": row["stock_id"],
            "signal_date": row["ngay"],
            "model_name": row["model"],
            "probability": float(row["probability"])
        })

    with engine.begin() as conn:
        conn.execute(sql, records)