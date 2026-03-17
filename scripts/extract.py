from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import DateTime, Text, text

from scripts.config import REQUIRED_COLUMNS, get_settings
from scripts.database import get_engine

logger = logging.getLogger(__name__)


def _read_demo_state(state_path: Path) -> dict[str, int | str | bool]:
    if not state_path.exists():
        return {}

    with state_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_demo_state(state_path: Path, state: dict[str, int | str | bool]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def extract_data(source_csv_path: str | Path | None = None) -> dict[str, int | str]:
    settings = get_settings()
    csv_path = Path(source_csv_path) if source_csv_path else settings.source_csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    logger.info("Reading Shopee source CSV from %s", csv_path)
    # Extract always starts from the source CSV and keeps only the columns needed downstream.
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in source CSV: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    if df.empty:
        raise ValueError("Source CSV is empty after selecting the required columns.")

    total_source_count = len(df)
    batch_size = max(1, settings.demo_batch_size)
    # The demo state allows each DAG run to continue from the next slice of the dataset.
    state = _read_demo_state(settings.demo_state_path)
    batch_start = int(state.get("next_offset", 0))
    cycle_number = int(state.get("cycle_number", 1))

    if batch_start >= total_source_count:
        logger.info("Reached end of dataset. Restarting demo from the first batch.")
        batch_start = 0
        cycle_number += 1

    batch_end = min(batch_start + batch_size, total_source_count)
    batch_df = df.iloc[batch_start:batch_end].copy()
    if batch_df.empty:
        raise ValueError("No rows available for the current demo batch.")

    extracted_count = len(batch_df)
    remaining_rows = total_source_count - batch_end
    batch_df["ingested_at"] = datetime.now(UTC).replace(tzinfo=None, microsecond=0)

    logger.info(
        "Extracted demo batch %s for cycle %s: rows %s to %s (%s records, %s remaining)",
        int(state.get("run_count", 0)) + 1,
        cycle_number,
        batch_start,
        batch_end - 1,
        extracted_count,
        remaining_rows,
    )
    engine = get_engine(settings)

    # Raw data is loaded as-is into the raw table so transform can work from a stable staging layer.
    batch_df.to_sql(
        name=settings.raw_table,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=2_000,
        method="multi",
        dtype={
            "id": Text(),
            "title": Text(),
            "seller_name": Text(),
            "item_category_detail": Text(),
            "delivery": Text(),
            "price_ori": Text(),
            "price_actual": Text(),
            "item_rating": Text(),
            "total_rating": Text(),
            "total_sold": Text(),
            "favorite": Text(),
            "w_date": Text(),
            "timestamp": Text(),
            "link_ori": Text(),
            "sitename": Text(),
            "ingested_at": DateTime(),
        },
    )

    with engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_{settings.raw_table}_id "
                f"ON {settings.raw_table} (id)"
            )
        )

    next_offset = batch_end
    wrapped_for_next_run = next_offset >= total_source_count
    # Persist batch progress so the next trigger continues from the correct offset.
    _write_demo_state(
        settings.demo_state_path,
        {
            "next_offset": next_offset,
            "total_rows": total_source_count,
            "batch_size": batch_size,
            "last_batch_start": batch_start,
            "last_batch_end": batch_end,
            "last_batch_size": extracted_count,
            "remaining_rows": remaining_rows,
            "cycle_number": cycle_number,
            "run_count": int(state.get("run_count", 0)) + 1,
            "wrapped_for_next_run": wrapped_for_next_run,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    logger.info(
        "Loaded %s raw records into PostgreSQL table %s",
        extracted_count,
        settings.raw_table,
    )
    return {
        "source_path": str(csv_path),
        "state_path": str(settings.demo_state_path),
        "total_source_count": total_source_count,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "remaining_rows": remaining_rows,
        "cycle_number": cycle_number,
        "wrapped_for_next_run": wrapped_for_next_run,
        "record_count": extracted_count,
        "table_name": settings.raw_table,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(extract_data())
