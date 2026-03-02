# 📘 Tài liệu mô tả phần mềm BankGuard AI
## Hệ thống Giám sát Toàn diện Ngân hàng (Comprehensive Banking Monitoring System)

**Phiên bản:** 1.0  
**Ngày phát hành:** 02/03/2026  
**Đơn vị phát triển:** BankGuard AI Team – KTNN  

---

## 📑 Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc phần mềm](#2-kiến-trúc-phần-mềm)
3. [Luồng xử lý dữ liệu (Data Pipeline)](#3-luồng-xử-lý-dữ-liệu-data-pipeline)
4. [Các mô hình học máy (Machine Learning Models)](#4-các-mô-hình-học-máy-machine-learning-models)
5. [Các nhóm chỉ số giám sát (Feature Groups)](#5-các-nhóm-chỉ-số-giám-sát-feature-groups)
6. [Mô tả chi tiết các tính năng Dashboard](#6-mô-tả-chi-tiết-các-tính-năng-dashboard)
7. [Công nghệ sử dụng](#7-công-nghệ-sử-dụng)

---

## 1. Tổng quan hệ thống

**BankGuard AI** là hệ thống giám sát sức khỏe tài chính ngân hàng dựa trên trí tuệ nhân tạo, sử dụng các thuật toán **học máy không giám sát (Unsupervised Machine Learning)** để:

- **Phát hiện bất thường (Anomaly Detection):** Tự động nhận diện các ngân hàng có chỉ số tài chính lệch chuẩn so với hệ thống.
- **Phân cụm rủi ro (Risk Clustering):** Phân loại ngân hàng thành 3 nhóm rủi ro (Thấp / Trung bình / Cao) dựa trên 20 chiều chỉ số.
- **Giải thích nguyên nhân (Explainability – XAI):** Xác định chỉ số nào gây ra bất thường cho từng ngân hàng cụ thể.
- **Phân tích ngoại bảng (Off-Balance Sheet Analysis):** Đánh giá mức độ rủi ro từ hoạt động phái sinh, bảo lãnh, cam kết tín dụng.

Hệ thống phân tích **20 chỉ số tài chính** (features) thuộc **3 nhóm rủi ro** trên bộ dữ liệu chuỗi thời gian của các ngân hàng.

---

## 2. Kiến trúc phần mềm

```
/comprehensive-banking-monitoring-system
│
├── app.py                          # Ứng dụng Dashboard chính (Streamlit)
├── config.py                       # Cấu hình tập trung: danh sách chỉ số, tham số mô hình
├── requirements.txt                # Danh sách thư viện Python
├── INSTRUCTIONS.md                 # Hướng dẫn dự án
│
├── models/
│   └── anomaly_detector.py         # Engine phân tích ML (Isolation Forest, K-Means, XAI)
│
├── utils/
│   └── data_processor.py           # Pipeline xử lý dữ liệu (Load → Impute → Scale)
│
└── data/
    └── time_series_dataset_enriched_v2.csv   # Dữ liệu đầu vào (chuỗi thời gian ngân hàng)
```

| Thành phần | Vai trò |
|---|---|
| `config.py` | Nguồn cấu hình duy nhất (Single Source of Truth) cho tên cột, nhóm features, tham số mô hình |
| `data_processor.py` | Đọc CSV → Xử lý giá trị thiếu (Median Imputation) → Tạo chỉ số phái sinh → Chuẩn hóa dữ liệu (StandardScaler) |
| `anomaly_detector.py` | Huấn luyện Isolation Forest + K-Means → Gán nhãn bất thường → Phân cụm rủi ro → Giải thích nguyên nhân |
| `app.py` | Dashboard tương tác 5 tab với biểu đồ Plotly, bộ lọc thời gian/vùng miền |

---

## 3. Luồng xử lý dữ liệu (Data Pipeline)

### 3.1 Quy trình từ dữ liệu thô đến kết quả phân tích

```
CSV Raw Data
    │
    ▼
[1] Load Data ──────────── Đọc file CSV (200 quan sát × 66 cột)
    │
    ▼
[2] Impute Missing ─────── Median Imputation cho các cột số
    │
    ▼
[3] Validate Features ──── Kiểm tra đủ 20 cột ML bắt buộc
    │
    ▼
[4] Feature Engineering ── Tạo chỉ số risk_to_profit_ratio = NPL / ROA
    │
    ▼
[5] StandardScaler ─────── Chuẩn hóa từng nhóm (3 nhóm × riêng từng scaler)
    │
    ▼
[6] Isolation Forest ───── Phát hiện bất thường (anomaly_score, is_anomaly)
    │
    ▼
[7] K-Means ────────────── Phân cụm 3 nhóm rủi ro + Sector DNA Profiling
    │
    ▼
[8] XAI Driver Analysis ── Xác định chỉ số gây bất thường (Z-score deviation)
    │
    ▼
[9] OBS Risk Scoring ───── Đánh giá rủi ro ngoại bảng
    │
    ▼
[Output] ───────────────── DataFrame 200 × 75 cột → Dashboard 5 tab
```

### 3.2 Chuẩn hóa dữ liệu (Scaling)

Hệ thống sử dụng **StandardScaler** (từ scikit-learn) theo từng nhóm chỉ số:

| Nhóm | Số features | Phương pháp |
|---|---|---|
| Financial Health | 9 | StandardScaler riêng |
| Concentration Risk | 5 | StandardScaler riêng |
| Exposure & Liquidity | 6 | StandardScaler riêng |

**Công thức chuẩn hóa:**

$$z = \frac{x - \mu}{\sigma}$$

Trong đó $\mu$ là trung bình và $\sigma$ là độ lệch chuẩn của từng cột trong nhóm.

---

## 4. Các mô hình học máy (Machine Learning Models)

### 4.1 Isolation Forest — Phát hiện bất thường

| Thuộc tính | Giá trị |
|---|---|
| **Loại mô hình** | Unsupervised Anomaly Detection |
| **Thư viện** | `sklearn.ensemble.IsolationForest` |
| **Số chiều đầu vào** | 20 features (đã chuẩn hóa) |
| **Tỷ lệ bất thường (contamination)** | 0.05 (5%) |
| **Số cây quyết định (n_estimators)** | 300 |
| **max_features** | `min(1.0, 10/20) = 0.5` |
| **random_state** | 42 |

#### Nguyên lý hoạt động:

**Isolation Forest** là thuật toán phát hiện bất thường dựa trên ý tưởng: *các điểm dữ liệu bất thường dễ bị "cô lập" hơn so với điểm bình thường*.

1. Thuật toán xây dựng **300 cây quyết định ngẫu nhiên** (Isolation Trees).
2. Mỗi cây chọn ngẫu nhiên một feature và một ngưỡng cắt để phân chia dữ liệu.
3. Các điểm bất thường nằm xa phân phối chung → cần **ít lần cắt hơn** để bị cô lập → **đường đi trong cây ngắn hơn**.
4. **Anomaly Score** được tính dựa trên độ sâu trung bình qua 300 cây:
   - Score **< 0** (âm): Bất thường (anomaly) — càng âm càng bất thường.
   - Score **> 0** (dương): Bình thường.

#### Kết quả đầu ra:

| Cột | Ý nghĩa |
|---|---|
| `anomaly_score` | Điểm bất thường (số thực, càng âm càng bất thường) |
| `is_anomaly` | Nhãn: `-1` = Bất thường, `1` = Bình thường |

#### Ưu điểm khi áp dụng cho giám sát ngân hàng:

- **Không cần nhãn (unsupervised):** Phù hợp khi không có dữ liệu lịch sử đã gán nhãn "ngân hàng có vấn đề".
- **Đa chiều (20 features):** Phát hiện bất thường phức tạp mà con người khó nhìn thấy khi phân tích từng chỉ số riêng lẻ.
- **Hiệu suất cao:** Xử lý nhanh với dữ liệu lớn nhờ cơ chế random subsampling.

---

### 4.2 K-Means Clustering — Phân cụm rủi ro

| Thuộc tính | Giá trị |
|---|---|
| **Loại mô hình** | Unsupervised Clustering |
| **Thư viện** | `sklearn.cluster.KMeans` |
| **Số chiều đầu vào** | 20 features (đã chuẩn hóa) |
| **Số cụm (n_clusters)** | 3 |
| **n_init** | 10 (chạy 10 lần với centroid khác nhau, chọn kết quả tốt nhất) |
| **max_iter** | 300 |
| **random_state** | 42 |

#### Nguyên lý hoạt động:

1. Khởi tạo **3 tâm cụm (centroids)** ngẫu nhiên trong không gian 20 chiều.
2. **Lặp lại** (tối đa 300 lần):
   - Gán mỗi ngân hàng vào cụm có tâm gần nhất (khoảng cách Euclidean).
   - Cập nhật tâm cụm = trung bình các điểm trong cụm.
3. Dừng khi tâm cụm không thay đổi.

#### Gán nhãn rủi ro (Data-Driven Risk Labeling):

Sau khi K-Means phân 3 cụm, hệ thống **tự động gán nhãn rủi ro** dựa trên chỉ số tổng hợp:

$$\text{risk\_composite} = \text{NPL Ratio trung bình cụm} - \text{CAR trung bình cụm}$$

- Cụm có `risk_composite` **thấp nhất** → **Low Risk** (Rủi ro thấp)
- Cụm có `risk_composite` **trung bình** → **Medium Risk** (Rủi ro trung bình)
- Cụm có `risk_composite` **cao nhất** → **High Risk** (Rủi ro cao)

#### Sector DNA Profiling:

Mỗi cụm được phân tích thêm cấu trúc tín dụng theo ngành:

| Ngành | Tên cột dữ liệu |
|---|---|
| Năng lượng | `sector_loans_energy` |
| Bất động sản | `sector_loans_real_estate` |
| Xây dựng | `sector_loans_construction` |
| Dịch vụ | `sector_loans_services` |
| Nông nghiệp | `sector_loans_agriculture` |

Ví dụ kết quả DNA:
- **Low Risk:** High Real Estate Exposure (36.3%) + Energy (23.7%)
- **Medium Risk:** High Real Estate Exposure (33.6%) + Construction (23.1%)
- **High Risk:** High Agriculture Exposure (33.1%)

#### Kết quả đầu ra:

| Cột | Ý nghĩa |
|---|---|
| `cluster_label` | Nhãn cụm: "Low Risk" / "Medium Risk" / "High Risk" |
| `cluster_dna` | Mô tả DNA ngành nghề đặc trưng của cụm |

---

### 4.3 Explainability (XAI) — Giải thích nguyên nhân bất thường

Đây **không phải** một mô hình ML riêng, mà là module phân tích hậu mô hình (post-hoc analysis) dựa trên **Z-Score Deviation**.

#### Nguyên lý:

Với mỗi ngân hàng bị đánh dấu bất thường (`is_anomaly = -1`):

1. Tính **median** và **độ lệch chuẩn** hệ thống cho tất cả 20 features.
2. Tính **Z-score tuyệt đối** cho từng feature:

$$|z_i| = \frac{|x_i - \text{median}_i|}{\sigma_i}$$

3. Feature có $|z_i|$ lớn nhất → **Primary Anomaly Driver** (nguyên nhân chính gây bất thường).
4. Feature đó thuộc nhóm nào → **Anomaly Driver Group**.

#### Kết quả đầu ra:

| Cột | Ý nghĩa |
|---|---|
| `anomaly_driver` | Tên chỉ số gây bất thường chính (ví dụ: `return_on_equity`) |
| `anomaly_driver_group` | Nhóm rủi ro chứa chỉ số đó (ví dụ: "Financial Health") |

---

### 4.4 OBS Risk Contribution — Đánh giá rủi ro ngoại bảng

Module đánh giá bổ sung mức độ rủi ro từ hoạt động ngoại bảng (Off-Balance Sheet) cho các ngân hàng bất thường.

#### Nguyên lý:

1. Sử dụng cột `obs_risk_indicator` (chỉ số rủi ro ngoại bảng tổng hợp).
2. Tính **ngưỡng phân vị 75%** (P75) của toàn hệ thống.
3. Ngân hàng bất thường có `obs_risk_indicator > P75` → **"High OBS Risk"**.
4. Tính Z-score cho từng ngân hàng:

$$z_{obs} = \frac{\text{obs\_risk\_indicator} - \text{median}}{\sigma}$$

#### Kết quả đầu ra:

| Cột | Ý nghĩa |
|---|---|
| `obs_risk_flag` | "High OBS Risk" hoặc "Normal OBS" |
| `obs_risk_zscore` | Z-score rủi ro ngoại bảng |

---

## 5. Các nhóm chỉ số giám sát (Feature Groups)

### 5.1 Nhóm Sức khỏe Tài chính (Financial Health) — 9 chỉ số

| # | Tên chỉ số | Mô tả | Ý nghĩa giám sát |
|---|---|---|---|
| 1 | `capital_adequacy_ratio` | Tỷ lệ an toàn vốn (CAR) | CAR thấp → vốn không đủ bù đắp rủi ro |
| 2 | `npl_ratio` | Tỷ lệ nợ xấu (NPL) | NPL cao → chất lượng tín dụng kém |
| 3 | `liquidity_coverage_ratio` | Tỷ lệ bao phủ thanh khoản (LCR) | LCR thấp → rủi ro thanh khoản ngắn hạn |
| 4 | `nsfr` | Tỷ lệ ổn định tài trợ ròng | NSFR thấp → tài trợ dài hạn không ổn định |
| 5 | `provision_coverage_ratio` | Tỷ lệ bao phủ dự phòng | Thấp → chưa trích lập đủ dự phòng cho nợ xấu |
| 6 | `loan_to_deposit_ratio` | Tỷ lệ cho vay / huy động | Quá cao → rủi ro thanh khoản |
| 7 | `return_on_assets` | Lợi nhuận trên tổng tài sản (ROA) | Thấp/âm → hiệu quả kinh doanh kém |
| 8 | `return_on_equity` | Lợi nhuận trên vốn chủ sở hữu (ROE) | Thấp/âm → vốn chủ sở hữu sinh lời kém |
| 9 | `net_interest_margin` | Biên lãi ròng | Thấp → khả năng sinh lời từ hoạt động tín dụng yếu |

### 5.2 Nhóm Tập trung Rủi ro (Concentration Risk) — 5 chỉ số

| # | Tên chỉ số | Mô tả | Ý nghĩa giám sát |
|---|---|---|---|
| 1 | `sector_concentration_hhi` | Chỉ số HHI tập trung ngành | HHI cao → tín dụng tập trung vào ít ngành |
| 2 | `top20_borrower_concentration` | Tỷ lệ tập trung 20 khách hàng vay lớn nhất | Cao → rủi ro tập trung khách hàng vay |
| 3 | `geographic_concentration` | Mức tập trung địa lý | Cao → hoạt động phụ thuộc vào ít khu vực |
| 4 | `top20_depositors_ratio` | Tỷ lệ 20 người gửi tiền lớn nhất | Cao → rủi ro rút tiền hàng loạt |
| 5 | `top5_depositors_ratio` | Tỷ lệ 5 người gửi tiền lớn nhất | Cao → phụ thuộc nguồn vốn lớn |

### 5.3 Nhóm Ngoại bảng & Thanh khoản (Exposure & Liquidity) — 6 chỉ số

| # | Tên chỉ số | Mô tả | Ý nghĩa giám sát |
|---|---|---|---|
| 1 | `derivatives_to_assets_ratio` | Tỷ lệ phái sinh / tài sản | Cao → rủi ro từ giao dịch phái sinh lớn |
| 2 | `unused_lines_to_loans_ratio` | Hạn mức tín dụng chưa sử dụng / dư nợ | Cao → nghĩa vụ tiềm ẩn lớn |
| 3 | `guarantees_to_loans_ratio` | Bảo lãnh / dư nợ | Cao → nghĩa vụ ngoại bảng lớn |
| 4 | `obs_to_assets_ratio` | Hoạt động ngoại bảng / tổng tài sản | Cao → rủi ro ngoại bảng lớn |
| 5 | `wholesale_dependency_ratio` | Mức phụ thuộc vốn bán buôn | Cao → nguồn vốn không ổn định |
| 6 | `liquidity_concentration_risk` | Rủi ro tập trung thanh khoản | Cao → thanh khoản phụ thuộc ít nguồn |

### 5.4 Nhóm dữ liệu ngành (Sector Loans) — 5 cột

| Cột | Ngành |
|---|---|
| `sector_loans_energy` | Năng lượng |
| `sector_loans_real_estate` | Bất động sản |
| `sector_loans_construction` | Xây dựng |
| `sector_loans_services` | Dịch vụ |
| `sector_loans_agriculture` | Nông nghiệp |

### 5.5 Chỉ số phái sinh (Derived Feature)

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| `risk_to_profit_ratio` | NPL Ratio / ROA | Tỷ lệ rủi ro so với khả năng sinh lời — cao → rủi ro lớn hơn lợi nhuận |

---

## 6. Mô tả chi tiết các tính năng Dashboard

### 6.1 Thanh bên (Sidebar) — Bộ lọc & Điều khiển

| Tính năng | Mô tả |
|---|---|
| **Nút "Run Analysis"** | Xóa cache và chạy lại toàn bộ pipeline ML |
| **Bộ lọc Period** | Chọn một hoặc nhiều kỳ báo cáo |
| **Bộ lọc Region** | Lọc theo vùng miền |
| **Bộ lọc Bank Type** | Lọc theo loại ngân hàng |

---

### 6.2 Tab 1 — 📊 System Overview (Tổng quan hệ thống)

Cung cấp cái nhìn toàn cảnh về tình trạng sức khỏe toàn hệ thống.

| Thành phần | Loại biểu đồ | Mô tả |
|---|---|---|
| **KPI Cards** (5 thẻ) | Metric | Tổng tài sản giám sát, NPL hệ thống, Số bất thường, CAR trung bình, Số ngân hàng OBS rủi ro cao |
| **Interactive Risk Map** | Scatter Plot (Plotly) | Trục X = CAR, Trục Y = NPL, Kích thước = Tổng tài sản, Màu = Cụm rủi ro |
| **Cluster Distribution & DNA** | Pie Chart (Donut) | Tỷ lệ phân bổ 3 cụm rủi ro + mô tả DNA ngành nghề |
| **Anomaly Driver Group Distribution** | Bar Chart | Phân bổ nguyên nhân bất thường theo nhóm rủi ro |

---

### 6.3 Tab 2 — 🚨 Anomaly Alerts (Cảnh báo bất thường)

Trung tâm cảnh báo với bảng dữ liệu chi tiết.

| Thành phần | Mô tả |
|---|---|
| **KPI Cards** (4 thẻ) | Tổng quan sát, Số bất thường, Tỷ lệ bất thường, Số OBS rủi ro cao |
| **Bảng dữ liệu** | Danh sách tất cả ngân hàng, sắp xếp theo anomaly_score. Hàng bất thường tô đỏ |
| **Download CSV** | Tải xuống báo cáo bất thường dạng CSV |

**Các cột hiển thị:** bank_id, period, npl_ratio, capital_adequacy_ratio, liquidity_coverage_ratio, anomaly_score, is_anomaly, cluster_label, cluster_dna, anomaly_driver, anomaly_driver_group, obs_risk_flag, obs_risk_zscore.

---

### 6.4 Tab 3 — 🔍 Individual Deep-Dive (Phân tích chi tiết từng ngân hàng)

Cho phép kiểm tra sâu một ngân hàng cụ thể so với trung bình hệ thống.

| Thành phần | Loại biểu đồ | Mô tả |
|---|---|---|
| **Thẻ thông tin** (5 thẻ) | Metric | Bank ID, Region, Cluster, Anomaly Status, OBS Risk |
| **Sector DNA** | Info box | Mô tả DNA ngành nghề của cụm |
| **Primary Anomaly Driver** | Warning box | Chỉ số gây bất thường chính + nhóm rủi ro |
| **Radar Charts** (3 biểu đồ) | Radar/Spider | So sánh ngân hàng vs hệ thống cho từng nhóm: Financial Health (9), Concentration Risk (5), Exposure & Liquidity (6) |
| **Risk Factor Breakdown** | Horizontal Bar | Top 10 chỉ số có Z-score deviation lớn nhất (trên 20 features) |
| **Historical Trend** | Line Chart | Diễn biến NPL, CAR, LCR, NIM theo thời gian |
| **Download CSV** | Button | Tải toàn bộ dữ liệu đã lọc |

---

### 6.5 Tab 4 — 🏭 Sectoral Analysis (Phân tích ngành)

Phân tích cơ cấu tín dụng theo 5 ngành kinh tế.

| Thành phần | Loại biểu đồ | Mô tả |
|---|---|---|
| **Lending Structure by Bank** | Stacked Bar Chart | Cấu trúc cho vay theo ngành của từng ngân hàng (kỳ mới nhất) |
| **Sector Proportion Heatmap** | Heatmap | Tỷ lệ % cho vay mỗi ngành so với tổng dư nợ – phát hiện tập trung tín dụng |
| **Concentration Radar** | Radar Chart | So sánh 5 chỉ số Concentration Risk của 1 ngân hàng vs hệ thống |

---

### 6.6 Tab 5 — 📋 Off-Balance Sheet (Phân tích ngoại bảng)

Đánh giá rủi ro từ hoạt động ngoại bảng: phái sinh, bảo lãnh, cam kết.

| Thành phần | Loại biểu đồ | Mô tả |
|---|---|---|
| **KPI Cards** (4 thẻ) | Metric | OBS/Assets trung bình, Wholesale Dependency, Derivatives/Assets, Số ngân hàng OBS rủi ro cao |
| **Exposure & Liquidity Radar** | Radar Chart | So sánh 6 chỉ số Exposure của 1 ngân hàng vs hệ thống |
| **Funding Stress Test** | Scatter Plot | Trục X = LCR, Trục Y = Wholesale Dependency – ngân hàng ở góc trên-trái = nguy hiểm (phụ thuộc vốn bán buôn cao + thanh khoản thấp) |
| **Derivatives Leverage Gauge** | Gauge Chart | Đồng hồ đo tỷ lệ phái sinh / tổng tài sản (%) với 3 mức: Xanh (<50%), Vàng (50-100%), Đỏ (>100%) |

---

## 7. Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản | Vai trò |
|---|---|---|---|
| Ngôn ngữ | Python | 3.x | Ngôn ngữ lập trình chính |
| Giao diện | Streamlit | 1.54.0 | Framework xây dựng dashboard tương tác |
| Trực quan hóa | Plotly | 6.5.2 | Biểu đồ tương tác (scatter, pie, radar, heatmap, gauge) |
| Xử lý dữ liệu | Pandas | 2.3.3 | Thao tác dữ liệu dạng bảng |
| Tính toán số | NumPy | 2.4.2 | Tính toán ma trận, thống kê |
| Học máy | scikit-learn | 1.8.0 | Isolation Forest, K-Means, StandardScaler |

---

### Tham số mô hình (từ `config.py`)

```python
MODEL_PARAMS = {
    'contamination': 0.05,   # Tỷ lệ bất thường dự kiến (5%)
    'n_clusters': 3,         # Số cụm rủi ro (Low / Medium / High)
    'random_state': 42       # Seed đảm bảo tái lập kết quả
}
```

---

## Tóm tắt mô hình học máy

| Mô hình | Mục đích | Loại | Đầu vào | Đầu ra |
|---|---|---|---|---|
| **Isolation Forest** | Phát hiện ngân hàng bất thường | Unsupervised Anomaly Detection | 20 features (scaled) | `anomaly_score`, `is_anomaly` |
| **K-Means (k=3)** | Phân loại mức rủi ro | Unsupervised Clustering | 20 features (scaled) | `cluster_label`, `cluster_dna` |
| **Z-Score Deviation (XAI)** | Giải thích nguyên nhân bất thường | Statistical Analysis | 20 features (original) | `anomaly_driver`, `anomaly_driver_group` |
| **OBS Risk Scoring** | Đánh giá rủi ro ngoại bảng | Statistical Threshold (P75) | `obs_risk_indicator` | `obs_risk_flag`, `obs_risk_zscore` |

---

*Tài liệu này được tạo tự động từ mã nguồn dự án BankGuard AI.*  
*© 2026 BankGuard AI – Kiểm toán Nhà nước (KTNN)*
