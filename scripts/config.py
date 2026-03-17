from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STAGING_DIR = DATA_DIR / "staging"
SOURCE_FILE = DATA_DIR / "shopee_sample_data.csv"
DEMO_STATE_FILE = STAGING_DIR / "shopee_demo_state.json"
DEFAULT_DEMO_BATCH_SIZE = 3_000

REQUIRED_COLUMNS = [
    "id",
    "title",
    "seller_name",
    "item_category_detail",
    "delivery",
    "price_ori",
    "price_actual",
    "item_rating",
    "total_rating",
    "total_sold",
    "favorite",
    "w_date",
    "timestamp",
    "link_ori",
    "sitename",
]

CLEAN_OUTPUT_COLUMNS = [
    "id",
    "site_name",
    "seller_name",
    "title",
    "category_path",
    "main_category",
    "delivery_region",
    "price_original",
    "price_actual",
    "discount_pct",
    "has_discount",
    "item_rating",
    "total_rating_count",
    "total_sold_count",
    "favorite_count",
    "title_length",
    "is_popular",
    "ml_popularity_score",
    "ml_predicted_popular",
    "snapshot_date",
    "scraped_at",
    "link_ori",
]


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    source_csv_path: Path
    staging_dir: Path
    demo_batch_size: int
    demo_state_path: Path
    raw_table: str = "shopee_raw"
    clean_table: str = "shopee_clean"

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def transformed_csv_path(self) -> Path:
        return self.staging_dir / "shopee_clean_transformed.csv"

    @property
    def model_artifact_path(self) -> Path:
        return self.staging_dir / "shopee_popularity_model.joblib"

    @property
    def metrics_artifact_path(self) -> Path:
        return self.staging_dir / "shopee_ml_metrics.json"


def get_settings() -> Settings:
    return Settings(
        db_host=os.getenv("WAREHOUSE_DB_HOST", "localhost"),
        db_port=int(os.getenv("WAREHOUSE_DB_PORT", "5432")),
        db_name=os.getenv("WAREHOUSE_DB_NAME", "shopee_warehouse"),
        db_user=os.getenv("WAREHOUSE_DB_USER", "airflow"),
        db_password=os.getenv("WAREHOUSE_DB_PASSWORD", "airflow"),
        source_csv_path=Path(os.getenv("SHOPEE_SOURCE_CSV", str(SOURCE_FILE))),
        staging_dir=Path(os.getenv("SHOPEE_STAGING_DIR", str(STAGING_DIR))),
        demo_batch_size=int(os.getenv("SHOPEE_DEMO_BATCH_SIZE", str(DEFAULT_DEMO_BATCH_SIZE))),
        demo_state_path=Path(os.getenv("SHOPEE_DEMO_STATE_PATH", str(DEMO_STATE_FILE))),
    )
