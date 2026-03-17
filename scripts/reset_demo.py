from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from sqlalchemy import text

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import get_settings
from scripts.database import get_engine

logger = logging.getLogger(__name__)


def _remove_file(file_path: Path) -> bool:
    if not file_path.exists():
        logger.info("Skip missing file: %s", file_path)
        return False

    file_path.unlink()
    logger.info("Removed file: %s", file_path)
    return True


def reset_demo(*, skip_db: bool = False, keep_artifacts: bool = False) -> dict[str, object]:
    settings = get_settings()
    removed_files: list[str] = []
    dropped_tables: list[str] = []

    if not skip_db:
        engine = get_engine(settings)
        tables_to_drop = [
            settings.raw_table,
            settings.clean_table,
            f"{settings.clean_table}_stage",
        ]
        with engine.begin() as connection:
            for table_name in tables_to_drop:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                dropped_tables.append(table_name)
                logger.info("Dropped table if exists: %s", table_name)

    files_to_remove = [settings.demo_state_path]
    if not keep_artifacts:
        files_to_remove.extend(
            [
                settings.transformed_csv_path,
                settings.model_artifact_path,
                settings.metrics_artifact_path,
            ]
        )

    for file_path in files_to_remove:
        if _remove_file(file_path):
            removed_files.append(str(file_path))

    summary = {
        "db_reset": not skip_db,
        "artifacts_removed": not keep_artifacts,
        "dropped_tables": dropped_tables,
        "removed_files": removed_files,
        "next_run_behavior": "The next DAG trigger will start again from the first batch.",
    }
    logger.info("Demo reset completed: %s", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset demo batch state and clean Shopee ETL demo tables.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Only reset local demo files, keep PostgreSQL tables untouched.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep transformed CSV, ML model, and metrics artifacts.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = _build_parser()
    arguments = parser.parse_args()
    print(
        json.dumps(
            reset_demo(
                skip_db=arguments.skip_db,
                keep_artifacts=arguments.keep_artifacts,
            ),
            indent=2,
        )
    )
