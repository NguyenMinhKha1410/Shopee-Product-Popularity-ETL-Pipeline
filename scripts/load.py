from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Boolean, Date, DateTime, Float, Integer, Text, text

from scripts.config import CLEAN_OUTPUT_COLUMNS, get_settings
from scripts.database import get_engine

logger = logging.getLogger(__name__)


def _table_dtypes() -> dict[str, object]:
    return {
        "id": Text(),
        "site_name": Text(),
        "seller_name": Text(),
        "title": Text(),
        "category_path": Text(),
        "main_category": Text(),
        "delivery_region": Text(),
        "price_original": Float(),
        "price_actual": Float(),
        "discount_pct": Float(),
        "has_discount": Boolean(),
        "item_rating": Float(),
        "total_rating_count": Integer(),
        "total_sold_count": Integer(),
        "favorite_count": Integer(),
        "title_length": Integer(),
        "is_popular": Boolean(),
        "ml_popularity_score": Float(),
        "ml_predicted_popular": Boolean(),
        "snapshot_date": Date(),
        "scraped_at": DateTime(),
        "link_ori": Text(),
    }


def _create_clean_table_sql(table_name: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            site_name TEXT NOT NULL,
            seller_name TEXT NOT NULL,
            title TEXT NOT NULL,
            category_path TEXT NOT NULL,
            main_category TEXT NOT NULL,
            delivery_region TEXT NOT NULL,
            price_original DOUBLE PRECISION NOT NULL,
            price_actual DOUBLE PRECISION NOT NULL CHECK (price_actual > 0),
            discount_pct DOUBLE PRECISION NOT NULL CHECK (discount_pct >= 0),
            has_discount BOOLEAN NOT NULL,
            item_rating DOUBLE PRECISION NOT NULL,
            total_rating_count INTEGER NOT NULL CHECK (total_rating_count >= 0),
            total_sold_count INTEGER NOT NULL CHECK (total_sold_count >= 0),
            favorite_count INTEGER NOT NULL CHECK (favorite_count >= 0),
            title_length INTEGER NOT NULL CHECK (title_length > 0),
            is_popular BOOLEAN NOT NULL,
            ml_popularity_score DOUBLE PRECISION NOT NULL CHECK (ml_popularity_score >= 0 AND ml_popularity_score <= 1),
            ml_predicted_popular BOOLEAN NOT NULL,
            snapshot_date DATE NOT NULL,
            scraped_at TIMESTAMP NOT NULL,
            link_ori TEXT NOT NULL
        )
    """


def load_data(
    transformed_csv_path: str | Path | None = None,
    *,
    reset_table: bool = False,
) -> dict[str, int | str]:
    settings = get_settings()
    csv_path = Path(transformed_csv_path) if transformed_csv_path else settings.transformed_csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Transformed file not found: {csv_path}")

    logger.info("Loading transformed dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Transformed dataset is empty. The load step cannot continue.")

    missing_columns = [column for column in CLEAN_OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in transformed CSV: {missing_columns}")

    float_columns = [
        "price_original",
        "price_actual",
        "discount_pct",
        "item_rating",
        "ml_popularity_score",
    ]
    int_columns = [
        "total_rating_count",
        "total_sold_count",
        "favorite_count",
        "title_length",
    ]
    bool_columns = ["has_discount", "is_popular", "ml_predicted_popular"]

    for column in float_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in int_columns:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    for column in bool_columns:
        df[column] = df[column].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="raise").dt.date
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="raise")

    if df["id"].duplicated().any():
        duplicate_count = int(df["id"].duplicated().sum())
        raise ValueError(f"Validation failed. Duplicate id values: {duplicate_count}")

    if (df["price_actual"] <= 0).any():
        raise ValueError("Validation failed. Found non-positive price_actual values.")

    if df.isna().any().any():
        null_summary = df.isna().sum()
        remaining_nulls = null_summary[null_summary > 0].to_dict()
        raise ValueError(f"Validation failed. Remaining NULL values: {remaining_nulls}")

    engine = get_engine(settings)
    # Bước Load dùng bảng stage trước để từng batch được kiểm tra và merge an toàn.
    stage_table = f"{settings.clean_table}_stage"
    batch_count = len(df)
    insert_columns = ", ".join(CLEAN_OUTPUT_COLUMNS)

    with engine.begin() as connection:
        if reset_table:
            # Khi vòng lặp demo quay lại batch 0, bảng clean sẽ được dựng lại từ đầu.
            logger.info("Resetting %s because the demo loop restarted from batch 0", settings.clean_table)
            connection.execute(text(f"DROP TABLE IF EXISTS {settings.clean_table}"))

        connection.execute(text(_create_clean_table_sql(settings.clean_table)))

    df = df[CLEAN_OUTPUT_COLUMNS].copy()
    df.to_sql(
        name=stage_table,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=2_000,
        method="multi",
        dtype=_table_dtypes(),
    )

    with engine.begin() as connection:
        # Xóa trước các id đã tồn tại để nhiều lần chạy có hành vi giống upsert theo id.
        connection.execute(
            text(
                f"DELETE FROM {settings.clean_table} AS target "
                f"USING {stage_table} AS stage "
                f"WHERE target.id = stage.id"
            )
        )
        connection.execute(
            text(
                f"INSERT INTO {settings.clean_table} ({insert_columns}) "
                f"SELECT {insert_columns} FROM {stage_table}"
            )
        )
        # Tạo index để các truy vấn demo và phân tích chạy nhanh hơn.
        connection.execute(text(f"DROP TABLE IF EXISTS {stage_table}"))
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{settings.clean_table}_main_category "
                f"ON {settings.clean_table} (main_category)"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{settings.clean_table}_is_popular "
                f"ON {settings.clean_table} (is_popular)"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{settings.clean_table}_seller_name "
                f"ON {settings.clean_table} (seller_name)"
            )
        )
        total_record_count = connection.execute(
            text(f"SELECT COUNT(*) FROM {settings.clean_table}")
        ).scalar_one()

    logger.info(
        "Loaded %s batch records into PostgreSQL table %s. Total accumulated records: %s",
        batch_count,
        settings.clean_table,
        total_record_count,
    )

    return {
        "batch_record_count": batch_count,
        "total_record_count": int(total_record_count),
        "table_name": settings.clean_table,
        "input_path": str(csv_path),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(load_data())
