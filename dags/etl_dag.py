from __future__ import annotations

from datetime import datetime
import logging
import sys
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow")
# Thêm thư mục gốc của project vào Python path để Airflow import được module trong scripts/.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract import extract_data
from scripts.load import load_data
from scripts.transform import transform_data

logger = logging.getLogger(__name__)


def run_extract() -> dict[str, int | str]:
    result = extract_data()
    logger.info("Extract task result: %s", result)
    return result


def run_transform(**context) -> dict[str, int | str]:
    # Lấy metadata của batch từ task extract để truyền tiếp cho các bước sau qua XCom.
    extract_result = context["ti"].xcom_pull(task_ids="extract_data") or {}
    result = transform_data()
    for key in [
        "state_path",
        "total_source_count",
        "batch_start",
        "batch_end",
        "remaining_rows",
        "cycle_number",
        "wrapped_for_next_run",
    ]:
        if key in extract_result:
            result[key] = extract_result[key]
    logger.info("Transform task result: %s", result)
    return result


def run_load(**context) -> dict[str, int | str]:
    transform_result = context["ti"].xcom_pull(task_ids="transform_data")
    if not transform_result or "output_path" not in transform_result:
        raise ValueError("Missing transformed file path from transform_data task.")

    # Nếu batch hiện tại bắt đầu từ 0 thì vòng demo đã quay lại đầu dataset và cần reset bảng clean.
    reset_table = int(transform_result.get("batch_start", 0)) == 0
    result = load_data(
        transform_result["output_path"],
        reset_table=reset_table,
    )
    logger.info("Load task result: %s", result)
    return result


with DAG(
    dag_id="etl_shopee_pipeline",
    description="ETL pipeline for Shopee product data with ML scoring",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["etl", "shopee", "postgres", "ml"],
) as dag:
    extract_task = PythonOperator(
        task_id="extract_data",
        python_callable=run_extract,
    )

    transform_task = PythonOperator(
        task_id="transform_data",
        python_callable=run_transform,
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=run_load,
    )

    # Dependency của DAG: Extract -> Transform -> Load.
    extract_task >> transform_task >> load_task
