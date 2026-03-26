# ETL Shopee Product Pipeline

## 1. Giới thiệu hệ thống

Project này đã được chuyển từ bài toán HR sang dataset Shopee. Pipeline đọc dữ liệu sản phẩm từ file `data/shopee_sample_data.csv`, sau đó:

- Load dữ liệu gốc vào bảng `shopee_raw`
- Clean và chuẩn hóa dữ liệu sản phẩm
- Tạo feature phục vụ phân tích bán hàng
- Huấn luyện một model ML nhỏ để chấm điểm mức độ `popular` của sản phẩm
- Load dữ liệu sạch vào bảng `shopee_clean`

Pipeline hiện chạy theo chế độ `demo batch`:

- Mỗi lần trigger DAG chỉ xử lý một phần dataset
- Phần còn lại được giữ lại cho các lần trigger tiếp theo
- Bảng `shopee_clean` sẽ tăng dần qua từng run để bạn demo kiểu incremental load

## 2. Kiến trúc hệ thống

Thành phần chính:

- `Apache Airflow`: điều phối DAG `etl_shopee_pipeline`
- `PostgreSQL`: lưu `shopee_raw` và `shopee_clean`
- `Docker Compose`: chạy PostgreSQL, Airflow Webserver, Airflow Scheduler
- `Python scripts`: `extract.py`, `transform.py`, `load.py`

Luồng xử lý:

1. `extract_data`
   Đọc file Shopee CSV và nạp dữ liệu raw vào `shopee_raw`
2. `transform_data`
   Parse giá, rating, sold, favorite; clean text; tạo feature bán hàng; train model Logistic Regression
3. `load_data`
   Load dữ liệu sạch và ML score vào `shopee_clean`, giữ lại dữ liệu các batch trước

## 3. Cấu trúc project

```text
project/
|-- dags/
|   `-- etl_dag.py
|-- data/
|   |-- shopee_sample_data.csv
|   `-- staging/
|-- logs/
|-- postgres/
|   `-- init/
|       `-- 01-create-shopee-db.sql
|-- scripts/
|   |-- __init__.py
|   |-- config.py
|   |-- database.py
|   |-- extract.py
|   |-- transform.py
|   `-- load.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## 4. Transform và ML đang áp dụng

### Làm sạch dữ liệu

- Giữ lại các cột chính phục vụ bán hàng và ML:
  `id`, `title`, `seller_name`, `item_category_detail`, `delivery`, `price_ori`, `price_actual`, `item_rating`, `total_rating`, `total_sold`, `favorite`, `w_date`, `timestamp`, `link_ori`, `sitename`
- Chuẩn hóa text cho `seller_name`, `delivery`, `sitename`, category path
- Parse các compact number như:
  `8.1k -> 8100`, `Favorite (21.5k -> 21500`
- Điền giá trị mặc định hợp lý cho text/metric bị thiếu
- Xóa duplicate toàn bộ dòng và duplicate theo `id`
- Loại bỏ dòng có giá không hợp lệ
- Loại bỏ dòng còn `NULL` sau transform

### Feature engineering

- `category_path`: category path đã chuẩn hóa
- `main_category`: category cấp cao nhất
- `discount_pct`: phần trăm giảm giá
- `has_discount`: sản phẩm có giảm giá hay không
- `title_length`: độ dài tiêu đề
- `total_rating_count`, `total_sold_count`, `favorite_count`: số nguyên đã clean
- `snapshot_date`: ngày snapshot từ `w_date`
- `scraped_at`: timestamp convert từ epoch milliseconds

### ML nhỏ trong transform

- Tạo nhãn `is_popular`:
  sản phẩm được xem là popular nếu `total_sold_count >= median(total_sold_count)`
- Feature dùng cho model:
  `price_actual`, `discount_pct`, `item_rating`, `total_rating_count`, `favorite_count`, `title_length`, `has_discount`
- Model:
  `LogisticRegression`
- Kết quả được ghi ra:
  - `ml_popularity_score`
  - `ml_predicted_popular`
- Artifact sinh ra ở `data/staging/`:
  - `shopee_popularity_model.joblib`
  - `shopee_ml_metrics.json`

## 5. Hướng dẫn chạy hệ thống

Yêu cầu:

- Đã cài Docker Desktop
- Port `8081` và `5434` còn trống

Demo batch mặc định:

- `SHOPEE_DEMO_BATCH_SIZE=3000`
- Nghĩa là mỗi lần trigger DAG sẽ nạp 3,000 dòng tiếp theo
- Có thể đổi số này trong [docker-compose.yml](D:/Administrator/Documents/Project%20Cuoi%20Ki%20Mindx/docker-compose.yml#L44)

Bước 1. Khởi tạo Airflow metadata và user admin:

```bash
docker compose up airflow-init
```

Bước 2. Chạy toàn bộ stack:

```bash
docker compose up --build -d
```

Bước 3. Truy cập Airflow UI:

- URL: [http://localhost:8081](http://localhost:8081)
- Username: `admin`
- Password: `admin`

## 6. Trigger DAG

1. Mở Airflow UI
2. Tìm DAG `etl_shopee_pipeline`
3. Nhấn `Trigger DAG`
4. Theo dõi 3 task:
   `extract_data -> transform_data -> load_data`

Hành vi demo:

- Run 1: nạp batch đầu tiên vào `shopee_clean`
- Run 2: nạp batch tiếp theo và giữ lại dữ liệu của run 1
- Run 3, 4, ...: tiếp tục nạp các batch còn lại
- Khi đi hết dataset, run tiếp theo sẽ quay lại batch đầu và reset lại bảng `shopee_clean` để bạn demo lại từ đầu

## 7. Kiểm tra dữ liệu PostgreSQL

```bash
docker compose exec postgres psql -U airflow -d shopee_warehouse
```

Ví dụ query:

```sql
SELECT COUNT(*) FROM shopee_raw;
SELECT COUNT(*) FROM shopee_clean;
SELECT id, title, total_sold_count, ml_popularity_score
FROM shopee_clean
LIMIT 10;
```

Query để xem số record tăng dần theo từng lần demo:

```sql
SELECT COUNT(*) AS current_total_rows
FROM shopee_clean;
```

## 8. Logging và kiểm soát lỗi

Pipeline sẽ log:

- số record extract được
- số record bị loại vì duplicate
- số record bị loại vì giá lỗi hoặc null
- accuracy và ROC AUC của model ML
- số record load thành công vào PostgreSQL

Pipeline sẽ raise error nếu:

- file CSV rỗng
- thiếu cột bắt buộc
- dữ liệu raw rỗng
- sau transform vẫn còn null
- `id` bị trùng trước khi load
