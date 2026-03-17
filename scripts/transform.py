from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re

from joblib import dump
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from scripts.config import CLEAN_OUTPUT_COLUMNS, get_settings
from scripts.database import get_engine

logger = logging.getLogger(__name__)


def _normalize_text(value: object, default: str = "unknown") -> str:
    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none", "n/a"}:
        return default
    return re.sub(r"\s+", " ", text_value).lower()


def _parse_compact_number(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default

    text_value = str(value).strip().lower().replace(",", "")
    if text_value in {"", "nan", "none", "n/a", "no ratings yet"}:
        return default

    match = re.search(r"([0-9]*\.?[0-9]+)\s*([kmb]?)", text_value)
    if not match:
        return default

    number = float(match.group(1))
    suffix = match.group(2)
    multiplier = {"": 1.0, "k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[suffix]
    return number * multiplier


def _extract_category_tokens(value: object) -> list[str]:
    tokens = [_normalize_text(token, default="") for token in str(value).split("|")]
    tokens = [token for token in tokens if token]
    if tokens and tokens[0] == "shopee":
        tokens = tokens[1:]
    return tokens or ["unknown"]


def transform_data(output_csv_path: str | Path | None = None) -> dict[str, int | str]:
    settings = get_settings()
    output_path = Path(output_csv_path) if output_csv_path else settings.transformed_csv_path

    engine = get_engine(settings)
    query = (
        f"SELECT id, title, seller_name, item_category_detail, delivery, "
        f"price_ori, price_actual, item_rating, total_rating, total_sold, "
        f"favorite, w_date, timestamp, link_ori, sitename "
        f"FROM {settings.raw_table}"
    )
    # Bước Transform luôn đọc từ bảng raw trong PostgreSQL do bước Extract tạo ra.
    df = pd.read_sql(text(query), con=engine)

    if df.empty:
        raise ValueError("Raw table is empty. The transform step cannot continue.")

    source_count = len(df)
    logger.info("Starting transform for %s records", source_count)

    # Xóa dòng trùng hoàn toàn trước, sau đó đảm bảo mỗi id sản phẩm chỉ còn một bản ghi.
    duplicate_rows_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    duplicate_id_removed = int(df.duplicated(subset=["id"]).sum())
    df = df.drop_duplicates(subset=["id"], keep="first").copy()

    df["id"] = df["id"].astype("string").str.strip().replace("", pd.NA)
    df["title"] = df["title"].astype("string").str.strip().replace("", pd.NA)
    df["seller_name"] = df["seller_name"].apply(_normalize_text)
    df["site_name"] = df["sitename"].apply(_normalize_text, default="shopee")
    df["delivery_region"] = df["delivery"].apply(_normalize_text)
    df["link_ori"] = df["link_ori"].astype("string").str.strip().replace("", pd.NA)

    # Tách category dạng phân cấp thành các cột dễ dùng cho phân tích.
    category_tokens = df["item_category_detail"].apply(_extract_category_tokens)
    df["main_category"] = category_tokens.str[0]
    df["category_path"] = category_tokens.apply(lambda tokens: " > ".join(tokens))

    # Chuyển các giá trị dạng text như "8.1k" về số để có thể validate và đưa vào model.
    df["price_original"] = df["price_ori"].apply(_parse_compact_number)
    df["price_actual"] = df["price_actual"].apply(_parse_compact_number)
    df["item_rating"] = df["item_rating"].apply(_parse_compact_number)
    df["total_rating_count"] = df["total_rating"].apply(_parse_compact_number).round().astype(int)
    df["total_sold_count"] = df["total_sold"].apply(_parse_compact_number).round().astype(int)
    df["favorite_count"] = df["favorite"].apply(_parse_compact_number).round().astype(int)

    invalid_price_mask = (df["price_actual"] <= 0) | (df["price_original"] < 0)
    invalid_price_removed = int(invalid_price_mask.sum())
    df = df.loc[~invalid_price_mask].copy()

    # Feature engineering: tạo thêm các đặc trưng như discount, độ dài tiêu đề và thời gian chuẩn hóa.
    df["price_original"] = df["price_original"].where(df["price_original"] > 0, df["price_actual"])
    df["discount_pct"] = (
        ((df["price_original"] - df["price_actual"]) / df["price_original"])
        .clip(lower=0)
        .mul(100)
        .round(2)
    )
    df["has_discount"] = df["discount_pct"].gt(0)
    df["title_length"] = df["title"].fillna("").str.len().astype(int)
    df["snapshot_date"] = pd.to_datetime(df["w_date"], errors="coerce").dt.date
    timestamp_ms = pd.to_numeric(df["timestamp"], errors="coerce")
    df["scraped_at"] = pd.to_datetime(timestamp_ms, unit="ms", errors="coerce", utc=True).dt.tz_localize(None)

    required_clean_columns = [
        "id",
        "title",
        "seller_name",
        "site_name",
        "category_path",
        "main_category",
        "delivery_region",
        "price_original",
        "price_actual",
        "item_rating",
        "total_rating_count",
        "total_sold_count",
        "favorite_count",
        "snapshot_date",
        "scraped_at",
        "link_ori",
    ]
    null_row_mask = df[required_clean_columns].isna().any(axis=1)
    null_rows_removed = int(null_row_mask.sum())
    df = df.loc[~null_row_mask].copy()

    if df.empty:
        raise ValueError("All records were removed during transformation.")

    popularity_threshold = float(df["total_sold_count"].median())
    # Nhãn huấn luyện được tạo từ batch hiện tại: sản phẩm bán cao hơn hoặc bằng median sẽ là "popular".
    df["is_popular"] = df["total_sold_count"].ge(popularity_threshold)

    feature_columns = [
        "price_actual",
        "discount_pct",
        "item_rating",
        "total_rating_count",
        "favorite_count",
        "title_length",
        "has_discount",
    ]
    model_features = df[feature_columns].copy()
    model_features["has_discount"] = model_features["has_discount"].astype(int)
    target = df["is_popular"].astype(int)

    model_name = "logistic_regression_popularity_classifier"
    if target.nunique() < 2:
        # Batch demo quá nhỏ có thể chỉ có một class, nên dùng model fallback để tránh pipeline bị fail.
        logger.warning(
            "Current batch contains only one popularity class. Falling back to DummyClassifier for demo stability."
        )
        constant_class = int(target.iloc[0])
        popularity_model = DummyClassifier(strategy="constant", constant=constant_class)
        popularity_model.fit(model_features, target)
        accuracy = 1.0
        roc_auc = None
        model_name = "dummy_classifier_popularity_fallback"
        df["ml_popularity_score"] = float(constant_class)
        df["ml_predicted_popular"] = bool(constant_class)
    else:
        # Huấn luyện model nhẹ để bảng clean vừa có dữ liệu business vừa có thêm điểm ML.
        X_train, X_test, y_train, y_test = train_test_split(
            model_features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )

        popularity_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1_000, random_state=42)),
            ]
        )
        popularity_model.fit(X_train, y_train)

        test_predictions = popularity_model.predict(X_test)
        test_probabilities = popularity_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, test_predictions)
        roc_auc = roc_auc_score(y_test, test_probabilities)

        df["ml_popularity_score"] = popularity_model.predict_proba(model_features)[:, 1].round(4)
        df["ml_predicted_popular"] = popularity_model.predict(model_features).astype(bool)
    df = df[CLEAN_OUTPUT_COLUMNS].copy()

    if df.isna().any().any():
        null_summary = df.isna().sum()
        remaining_nulls = null_summary[null_summary > 0].to_dict()
        raise ValueError(f"Validation failed. Remaining NULL values: {remaining_nulls}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Lưu dataset đã transform và artifact ML để bước Load và phần demo có thể sử dụng lại.
    df.to_csv(output_path, index=False)
    dump(popularity_model, settings.model_artifact_path)

    metrics = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "popularity_threshold": round(popularity_threshold, 2),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "record_count": int(len(df)),
    }
    settings.metrics_artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    final_count = len(df)
    removed_count = source_count - final_count

    logger.info("Transform completed successfully")
    logger.info("Source records: %s", source_count)
    logger.info("Clean records: %s", final_count)
    logger.info("Removed records: %s", removed_count)
    logger.info("Removed duplicate rows: %s", duplicate_rows_removed)
    logger.info("Removed duplicate ids: %s", duplicate_id_removed)
    logger.info("Removed invalid price rows: %s", invalid_price_removed)
    logger.info("Removed rows containing NULLs: %s", null_rows_removed)
    logger.info("Popularity model accuracy: %.4f", accuracy)
    if roc_auc is None:
        logger.info("Popularity model ROC AUC: not available for one-class batch fallback")
    else:
        logger.info("Popularity model ROC AUC: %.4f", roc_auc)

    return {
        "output_path": str(output_path),
        "model_path": str(settings.model_artifact_path),
        "metrics_path": str(settings.metrics_artifact_path),
        "source_count": source_count,
        "clean_count": final_count,
        "removed_count": removed_count,
        "duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_id_removed": duplicate_id_removed,
        "invalid_price_removed": invalid_price_removed,
        "null_rows_removed": null_rows_removed,
        "popularity_threshold": round(popularity_threshold, 2),
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "n/a",
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(transform_data())
