# 📊 BÁO CÁO PHÂN TÍCH DỰ ÁN — BankGuard AI 360°
# Hệ thống Giám sát Toàn diện Ngân hàng

> **Ngày phân tích:** 2026-03-03
> **Phiên bản:** 3.0 — Hybrid Rule-Based + Multi-Algorithm ML Ensemble + Multi-Model XAI
> **Đơn vị:** Kiểm toán Nhà nước (KTNN)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu
BankGuard AI 360° là nền tảng giám sát ngân hàng cấp sản xuất (production-grade), được thiết kế cho cơ quan giám sát và đội quản lý rủi ro nội bộ. Hệ thống nhập dữ liệu báo cáo định kỳ (CSV), đánh giá từng ngân hàng qua **hai lớp phân tích bổ trợ** và hiển thị kết quả qua dashboard tương tác Streamlit.

### 1.2 Năm mục tiêu cốt lõi
1. **Phát hiện bất thường (Anomaly Detection)** — Nhận diện ngân hàng có hồ sơ rủi ro bất thường qua 7 trụ cột rủi ro, sử dụng đồng thuận 3 thuật toán ML (Isolation Forest, LOF, One-Class SVM)
2. **Sàng lọc tuân thủ quy định (Regulatory Compliance)** — Gắn cờ vi phạm ngưỡng Basel III tức thì qua Expert Rule Engine (9 luật)
3. **Kết hợp rủi ro lai (Hybrid Risk Fusion)** — Tổng hợp ML consensus + rule violations → `Final_Hybrid_Risk_Status` (Critical / Warning / Normal)
4. **Giải thích được (Explainability)** — 3 phương pháp XAI (SHAP, Permutation Importance, Local Surrogate/LIME-style) cho cả 3 mô hình ML, xác định feature driver chính
5. **So sánh ngang hàng (Peer Benchmarking)** — Radar chart 360°, sector-loan DNA, phân tích xu hướng lịch sử

---

## 2. CÔNG NGHỆ SỬ DỤNG

| Thành phần | Công nghệ | Phiên bản | Vai trò |
|---|---|---|---|
| Ngôn ngữ | Python | 3.14.2 | Ngôn ngữ chính |
| Giao diện | Streamlit | 1.54.0 | Dashboard tương tác 5 tab |
| ML Engine | Scikit-learn | 1.8.0 | IF, LOF, SVM, K-Means, StandardScaler, permutation_importance, Ridge |
| Giải thích mô hình | SHAP | ≥0.46.0 | TreeExplainer (IF) + KernelExplainer (LOF, SVM) |
| Trực quan hóa | Plotly | 6.5.2 | Biểu đồ tương tác (radar, heatmap, scatter, bar, pie, waterfall, beeswarm) |
| Xử lý dữ liệu | Pandas / NumPy | 2.3.3 / 2.4.2 | DataFrame operations, numerical computing |
| Testing | pytest | 9.0.2 | Unit tests & integration tests (~120 tests) |

---

## 3. CẤU TRÚC DỰ ÁN

```
comprehensive banking monitoring system/
│
├── app.py                          # Dashboard Streamlit chính (1,449 dòng, 5 tab)
├── config.py                       # Cấu hình trung tâm (357 dòng)
├── requirements.txt                # Dependencies (6 packages)
│
├── models/
│   └── anomaly_detector.py         # ML Engine đa thuật toán + XAI (1,001 dòng)
│
├── utils/
│   └── data_processor.py           # Pipeline xử lý dữ liệu 6 giai đoạn (402 dòng)
│
├── tests/
│   ├── test_config.py              # Unit tests cho config (~230 dòng, 35 tests)
│   ├── test_data_processor.py      # Unit tests cho data processor (~305 dòng, 29 tests)
│   └── test_anomaly_detector.py    # Unit tests cho ML engine + XAI (618 dòng, ~56 tests)
│
├── data/
│   └── time_series_dataset_enriched_v2.csv  # Dữ liệu đầu vào (200 obs × 66 cột)
│
├── DOCUMENTATION.md                # Tài liệu kỹ thuật
├── NEW_DOCUMENTATION.md            # Tài liệu kỹ thuật v2
└── INSTRUCTIONS.md                 # Hướng dẫn sử dụng
```

**Tổng dòng code:** ~3,209 dòng (app.py: 1,449 + anomaly_detector.py: 1,001 + data_processor.py: 402 + config.py: 357)
**Tổng dòng test:** ~1,153 dòng (test_anomaly_detector.py: 618 + test_data_processor.py: 305 + test_config.py: 230)

---

## 4. KIẾN TRÚC HỆ THỐNG

### 4.1 Sơ đồ luồng dữ liệu tổng thể

```
CSV Input (200 obs × 66 cols)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  DATA PROCESSOR (utils/data_processor.py)        │
│  ┌──────────────────────────────────────────┐    │
│  │ Stage 1: Load CSV                        │    │
│  │ Stage 2: Median Imputation               │    │
│  │ Stage 3: Validate 26 ML Features         │    │
│  │ Stage 4: Feature Engineering              │    │
│  │   → risk_to_profit_ratio                 │    │
│  │   → efficiency_ratio                     │    │
│  │ Stage 5: Expert Rule Engine (9 luật)     │    │
│  │   → rule_violations, rule_risk_score     │    │
│  │ Stage 6: StandardScaler per Pillar       │    │
│  └──────────────────────────────────────────┘    │
│  Output: (df_processed, df_original, scalers)    │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  ML ENGINE (models/anomaly_detector.py)          │
│  ┌──────────────────────────────────────────┐    │
│  │ Step 1:  Pillar-by-Pillar Consensus      │    │
│  │   3 models × 7 pillars = 21 runs         │    │
│  │   → Consensus Score (0/33/66/100)         │    │
│  │ Step 2:  Overall ML Risk Score            │    │
│  │   → Mean of 7 consensus scores            │    │
│  │ Step 3:  Hybrid Fusion                    │    │
│  │   ML + Rules → Final Status               │    │
│  │ Step 4:  K-Means Clustering + DNA         │    │
│  │ Step 5:  SHAP (Global IF TreeExplainer)   │    │
│  │ Step 6:  Multi-Model SHAP (LOF, SVM)     │    │
│  │   → KernelExplainer + background sampling │    │
│  │ Step 7:  Permutation Feature Importance   │    │
│  │   → All 3 models                         │    │
│  │ Step 8:  Local Surrogate (LIME-style)     │    │
│  │   → Ridge regression trên perturbed data  │    │
│  │ Step 9:  Anomaly Driver Attribution       │    │
│  │ Step 10: OBS Risk Contribution            │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  DASHBOARD (app.py) — 5 Tabs                     │
│  Tab 1: Executive Summary                        │
│  Tab 2: Multi-Algo Comparison                    │
│  Tab 3: Risk Pillar Deep Dive                    │
│  Tab 4: 360° Bank Profiler                       │
│  Tab 5: XAI & Model Evaluation (Multi-Model)     │

### 4.2 Cơ chế Consensus Scoring (Bỏ phiếu đồng thuận)

Mỗi trụ cột rủi ro được đánh giá bởi 3 mô hình ML độc lập:

| Số mô hình gắn cờ | Consensus Score | Ý nghĩa |
|---|---|---|
| 3/3 | 100 | High Risk — Cả 3 mô hình đồng ý bất thường |
| 2/3 | 66 | Warning — Đa số đồng ý |
| 1/3 | 33 | Monitor — Chỉ 1 mô hình phát hiện |
| 0/3 | 0 | Normal — Không có bất thường |

### 4.3 Hybrid Risk Fusion (Kết hợp rủi ro lai)

```
Final_Hybrid_Risk_Status:
  ├── Critical: Overall_ML_Risk_Score ≥ 60  HOẶC  rule_risk_score ≥ 3
  ├── Warning:  Overall_ML_Risk_Score ≥ 30  HOẶC  rule_risk_score ≥ 1
  └── Normal:   Còn lại
```

---

## 5. PHÂN TÍCH CHI TIẾT TỪNG MODULE

### 5.1 config.py — Trung tâm Cấu hình (357 dòng)

**Vai trò:** Single source of truth cho toàn bộ hệ thống.

**Nội dung chính:**
- **IDENTIFIERS** (5 cột): bank_id, period, bank_type, region, external_credit_rating
- **7 Risk Pillars** (26 features ML):

| # | Trụ cột | Features | Mô tả |
|---|---|---|---|
| 1 | Credit Risk | 4 | npl_ratio, loan_growth_rate, provision_coverage_ratio, non_performing_loans |
| 2 | Liquidity Risk | 3 | liquidity_coverage_ratio, nsfr, net_cash_outflows_30d |
| 3 | Concentration Risk | 3 | sector_concentration_hhi, top20_borrower_concentration, geographic_concentration |
| 4 | Capital Adequacy | 3 | capital_adequacy_ratio, tier1_capital, risk_weighted_assets |
| 5 | Earnings & Efficiency | 5 | return_on_assets, return_on_equity, net_interest_margin, operating_expenses, operating_income |
| 6 | Off-Balance Sheet | 4 | obs_exposure_total, guarantees_issued, obs_to_assets_ratio, derivatives_to_assets_ratio |
| 7 | Funding Stability | 4 | wholesale_dependency_ratio, loan_to_deposit_ratio, top20_depositors_ratio, deposit_growth_rate |

- **Pillar 8** (Sector Loan Allocation): 5 cột sector_loans_* — chỉ dùng cho visualization/DNA profiling, không đưa vào ML
- **Expert Rules Engine** (9 luật Basel III):

| Rule ID | Cột | Điều kiện | Mức độ | Ý nghĩa |
|---|---|---|---|---|
| CAR_CRITICAL | capital_adequacy_ratio | < 8% | Critical | Vi phạm Basel III Pillar 1 |
| NPL_WARNING | npl_ratio | > 3% | Warning | Chất lượng tín dụng suy giảm |
| LCR_CRITICAL | liquidity_coverage_ratio | < 100% | Critical | Thiếu HQLA cho stress 30 ngày |
| LDR_WARNING | loan_to_deposit_ratio | > 100% | Warning | Cho vay vượt tiền gửi |
| HHI_HIGH_CONCENTRATION | sector_concentration_hhi | > 0.25 | Warning | Tập trung cho vay cao |
| NSFR_CRITICAL | nsfr | < 100% | Critical | Thiếu nguồn vốn ổn định (Basel III) |
| ROA_WARNING | return_on_assets | < 0% | Warning | Ngân hàng thua lỗ, lợi nhuận âm |
| WHOLESALE_DEPENDENCY_WARNING | wholesale_dependency_ratio | > 50% | Warning | Phụ thuộc quá mức vào vốn bán buôn |
| TOP20_BORROWER_CONCENTRATION_WARNING | top20_borrower_concentration | > 25% | Warning | Rủi ro tập trung tín dụng đơn lẻ cao |

- **ML Models Config** (3 mô hình):
  - **Isolation Forest**: contamination=0.05, n_estimators=300, max_features=0.5
  - **Local Outlier Factor**: n_neighbors=20, contamination=0.05, novelty=True
  - **One-Class SVM**: kernel=rbf, gamma=scale, nu=0.05

- **Pipeline Parameters**: contamination=0.05, n_clusters=3, ensemble_method=majority_vote
- **Feature Labels & Sector Labels**: Nhãn hiển thị cho dashboard

**Đánh giá config.py:**
- ✅ Tập trung hóa tốt — mọi thay đổi chỉ cần sửa 1 file
- ✅ 9 expert rules phủ rộng các trụ cột rủi ro chính (Basel III)
- ✅ Có backward-compatible aliases (FINANCIAL_HEALTH_FEATURES, EXPOSURE_LIQUIDITY_FEATURES)
- ✅ Có FEATURE_LABELS cho UI display

---

### 5.2 data_processor.py — Pipeline Xử lý Dữ liệu (402 dòng)

**Vai trò:** Chuyển đổi dữ liệu thô thành input sẵn sàng cho ML.

**6 Giai đoạn Pipeline:**

| Stage | Hàm | Mô tả |
|---|---|---|
| 1. Load | `load_data()` | Đọc CSV, kiểm tra file tồn tại và không rỗng |
| 2. Impute | `_impute_missing_values()` | Median imputation cho mọi cột số |
| 3. Validate | `_validate_features()` | Kiểm tra 26 ML features có đủ không |
| 4. Engineer | `_create_risk_to_profit_ratio()`, `_create_efficiency_ratio()` | Tạo 2 features phái sinh |
| 5. Rules | `evaluate_expert_rules()` | Đánh giá 9 luật Basel III → rule_violations, rule_risk_score |
| 6. Scale | `_scale_feature_group()` | StandardScaler per pillar (7 nhóm) |

**Output:** `(df_processed, df_original, scalers)`
- `df_processed`: Features đã scale cho ML
- `df_original`: Dữ liệu gốc + rule engine columns
- `scalers`: Dict[group_name → fitted StandardScaler]

**Đánh giá data_processor.py:**
- ✅ Pipeline rõ ràng, có logging chi tiết
- ✅ Vectorised rule evaluation (nhanh)
- ✅ Tách biệt df_processed và df_original
- ⚠️ Median imputation đơn giản — có thể cải thiện bằng KNN/MICE
- ⚠️ Chưa có outlier detection ở giai đoạn preprocessing
- ⚠️ Feature engineering chỉ tạo 2 features phái sinh (risk_to_profit_ratio, efficiency_ratio) nhưng không đưa vào ML features

---

### 5.3 anomaly_detector.py — ML Engine + XAI (1,001 dòng)

**Vai trò:** Trung tâm phân tích ML — Multi-Algorithm Consensus Scoring Engine + Multi-Model XAI.

**Class chính:** `BankAnomalyDetector`

**10 Bước trong `run_full_analysis()`:**

| Step | Method | Mô tả |
|---|---|---|
| 1 | `_run_pillar_consensus()` | 3 models × 7 pillars = 21 lần fit/predict |
| 2 | `_compute_overall_ml_score()` | Mean of 7 consensus scores |
| 3 | `_compute_hybrid_status()` | ML + Rules → Critical/Warning/Normal |
| 4 | `cluster_banks()` | K-Means (k=3) + Sector DNA profiling |
| 5 | `evaluate_model_with_shap()` | Global IF + SHAP TreeExplainer |
| 6 | `compute_multi_model_shap()` | LOF & SVM via SHAP KernelExplainer |
| 7 | `compute_permutation_importance()` | Permutation FI cho cả 3 models |
| 8 | `compute_local_surrogate()` | LIME-style (Ridge regression) cho anomalous banks |
| 9 | `get_anomaly_drivers()` | SHAP-based top-3 driver attribution |
| 10 | `compute_obs_risk_contribution()` | OBS risk via z-score |

**3 Phương pháp XAI (Explainable AI):**

| Phương pháp | Mô hình áp dụng | Kỹ thuật | Chi tiết |
|---|---|---|---|
| **SHAP** | IF (TreeExplainer), LOF & SVM (KernelExplainer) | Shapley values | Background sampling `shap.sample(X, 50)`, `nsamples=100` cho KernelExplainer |
| **Permutation Importance** | IF, LOF, SVM | `sklearn.inspection.permutation_importance` | `n_repeats=10`, model-agnostic, đo mức giảm performance khi shuffle feature |
| **Local Surrogate (LIME-style)** | IF, LOF, SVM | Custom implementation | Gaussian perturbation, exponential kernel weighting, Ridge regression, giới hạn 20 anomalous banks |

**Cơ chế K-Means Clustering:**
- Fit K-Means (k=3) trên 26 ML features
- Risk-label mapping dựa trên `risk_composite = npl_ratio - capital_adequacy_ratio`
- Sector DNA: Xác định sector cho vay chủ đạo của mỗi cluster

**Đánh giá anomaly_detector.py:**
- ✅ Consensus mechanism mạnh — giảm false positive
- ✅ 3 phương pháp XAI phủ cả 3 mô hình — explainability toàn diện
- ✅ SHAP TreeExplainer (nhanh, chính xác) cho IF + KernelExplainer (model-agnostic) cho LOF/SVM
- ✅ Local Surrogate tự implement — không phụ thuộc package `lime`
- ✅ Backward-compatible columns (is_anomaly, anomaly_score...)
- ✅ Robust error handling (try/except cho từng step)
- ⚠️ LOF với novelty=True fit rồi predict trên cùng data — không phải novelty detection thực sự
- ⚠️ K-Means risk labeling dựa trên npl_ratio - CAR có thể oversimplified
- ⚠️ KernelExplainer chậm hơn TreeExplainer — cần background sampling để tối ưu

---

### 5.4 app.py — Dashboard Streamlit (1,449 dòng)

**Vai trò:** Giao diện người dùng — command center 5 tab.

**5 Tab chính:**

| Tab | Tên | Nội dung chính |
|---|---|---|
| 1 | Executive Summary | 6 KPI cards, Hybrid Status pie, Risk Trajectory line chart, Cluster distribution, Driver attribution |
| 2 | Multi-Algo Comparison | Consensus heatmap (banks × pillars), Model Agreement Matrix (IF/LOF/SVM overlap), ML Score histogram |
| 3 | Risk Pillar Deep Dive | Pillar selector, 4 KPI per pillar, Feature scatter plots (outlier coloring), Feature statistics, CSV download |
| 4 | 360° Bank Profiler | Bank selector, Info bar, Rule violation alert, 7-pillar radar (bank vs peer), Per-pillar bar, Bank details, Funding pie, Sector pie, Historical trend |
| 5 | XAI & Model Evaluation | 3 phương pháp XAI với radio selector (xem chi tiết bên dưới) |

**Tab 5 — XAI & Model Evaluation (chi tiết):**

| Phương pháp | Nội dung hiển thị |
|---|---|
| **5A: SHAP (Multi-Model)** | Single model view (IF/LOF/SVM selector), Global feature importance bar, Beeswarm scatter, Per-bank SHAP waterfall, Cross-model comparison chart |
| **5B: Permutation Importance** | Cross-model comparison bar chart, Per-model individual importance with error bars |
| **5C: Local Surrogate (LIME-style)** | Per-bank coefficient chart, Top-5 risk drivers table, Cross-model local comparison |

**Tính năng nổi bật:**
- Sidebar filters: Period, Region, Bank Type, Credit Rating
- `@st.cache_data` cho data loading + ML analysis
- Custom CSS cho metric cards và alert boxes
- Peer benchmarking (so sánh bank vs bank_type average)
- CSV download cho từng pillar và full report
- `load_and_process()` trả về 5 giá trị: `(df_result, scalers, shap_values, shap_base_value, xai_artifacts)`

**Đánh giá app.py:**
- ✅ UI/UX chuyên nghiệp, layout responsive
- ✅ 5 tab phủ đầy đủ use cases
- ✅ Tab 5 XAI toàn diện — 3 phương pháp × 3 mô hình
- ✅ Peer benchmarking radar chart rất trực quan
- ✅ SHAP waterfall cho local explainability
- ⚠️ File dài (1,449 dòng) — nên tách thành components
- ⚠️ Một số hardcoded values (colors, thresholds)
- ⚠️ Chưa có error boundary cho từng tab

---

### 5.5 Dữ liệu đầu vào — time_series_dataset_enriched_v2.csv

| Thuộc tính | Giá trị |
|---|---|
| Số observations | 200 |
| Số cột | 66 |
| Ngân hàng | VCB, BIDV, CTG, ... (nhiều ngân hàng) |
| Kỳ báo cáo | Q1/2018 → Q3/2022 (theo quý) |
| Loại ngân hàng | large, medium, small |
| Vùng | north, south, central |
| Credit rating | A, BBB, ... |

**66 cột bao gồm:**
- 5 identifiers (bank_id, period, bank_type, stability, region)
- 17 cột tài chính gốc (total_assets, total_loans, NPL, provisions, deposits, equity, tier1, tier2, RWA, opex, income, net_income...)
- 26 ML features (7 pillars)
- 5 sector loan columns
- 13 cột bổ sung (top borrower, OBS, wholesale funding, derivatives...)


---

## 6. ĐÁNH GIÁ TỔNG THỂ

### 6.1 Điểm mạnh

| # | Điểm mạnh | Chi tiết |
|---|---|---|
| 1 | **Kiến trúc Hybrid** | Kết hợp Rule Engine (deterministic) + ML Ensemble (stochastic) giảm thiểu cả false positive và false negative |
| 2 | **Multi-Algorithm Consensus** | 3 thuật toán đa dạng (tree-based, density-based, boundary-based) bỏ phiếu đồng thuận — robust hơn single model |
| 3 | **Pillar-by-Pillar** | Phân tích riêng từng trụ cột rủi ro thay vì gộp chung — phát hiện rủi ro cụ thể hơn |
| 4 | **Multi-Model XAI** | 3 phương pháp XAI (SHAP, Permutation Importance, Local Surrogate) phủ cả 3 mô hình ML — explainability toàn diện |
| 5 | **Cấu hình tập trung** | config.py là single source of truth — dễ bảo trì, mở rộng |
| 6 | **Dashboard toàn diện** | 5 tab phủ từ executive overview đến per-bank deep dive |
| 7 | **Peer Benchmarking** | So sánh bank vs peer group — context-aware analysis |
| 8 | **Sector DNA Profiling** | Xác định đặc trưng cho vay của từng cluster rủi ro |
| 9 | **9 Expert Rules** | Phủ rộng Basel III: CAR, NPL, LCR, LDR, HHI, NSFR, ROA, Wholesale Dependency, Top-20 Borrower |
| 10 | **Unit Test Suite** | ~120 tests (pytest) phủ config, data processor, ML engine + XAI — giảm rủi ro regression |

### 6.2 Điểm yếu & Hạn chế

| # | Hạn chế | Mức độ | Giải thích |
|---|---|---|---|
| 1 | **Monolithic Architecture** | 🟡 Trung bình | Toàn bộ là 1 app Streamlit, chưa tách microservices |
| 2 | **Không có Database** | 🟡 Trung bình | Đọc trực tiếp từ CSV — không scale được |
| 3 | **Không có Authentication** | 🔴 Cao | Dashboard không có login/phân quyền |
| 4 | **LOF Novelty Mode** | 🟡 Trung bình | fit + predict trên cùng data không phải novelty detection thực sự |
| 5 | **app.py quá dài** | 🟡 Trung bình | 1,449 dòng trong 1 file — khó maintain |
| 6 | **Không có CI/CD** | 🟡 Trung bình | Không có pipeline tự động |
| 7 | **Dữ liệu nhỏ** | 🟡 Trung bình | 200 observations — ML models có thể overfit |
| 8 | **Không có Model Versioning** | 🟡 Trung bình | Models retrain mỗi lần chạy, không lưu trữ |

---

## 7. ĐỀ XUẤT HƯỚNG PHÁT TRIỂN

### 7.1 Ngắn hạn (1-2 tháng)

1. ~~**Thêm Unit Tests**~~ ✅ ĐÃ HOÀN THÀNH — 3 file test, ~120 tests (pytest), phủ config, data processor, ML engine + XAI
2. ~~**Mở rộng Expert Rules**~~ ✅ ĐÃ HOÀN THÀNH — Đã thêm NSFR < 100%, ROA < 0%, wholesale_dependency > 50%, top20_borrower_concentration > 25% (tổng 9 rules)
3. ~~**Thêm XAI đa mô hình**~~ ✅ ĐÃ HOÀN THÀNH — 3 phương pháp XAI (SHAP, Permutation Importance, Local Surrogate) cho cả 3 mô hình ML
4. **Tách app.py** — Chia thành components/tabs riêng biệt
5. **Thêm Authentication** — Streamlit Authenticator hoặc OAuth2
6. **Logging cải thiện** — Structured logging (JSON) thay vì print statements trong data_processor

### 7.2 Trung hạn (3-6 tháng)

7. **Database Backend** — PostgreSQL/TimescaleDB thay CSV cho time-series data
8. **API Layer** — FastAPI REST API tách biệt ML engine khỏi UI
9. **Model Registry** — MLflow để track model versions, parameters, metrics
10. **Automated Data Ingestion** — Scheduled pipeline (Airflow/Prefect) thay vì manual CSV upload
11. **Thêm Supervised Learning** — Train classification model trên historical labels (nếu có)

### 7.3 Dài hạn (6-12 tháng)

12. **Microservices Architecture** — Tách Data Processor, ML Engine, Dashboard thành services riêng
13. **Real-time Monitoring** — Kafka/streaming cho near-real-time anomaly detection
14. **Multi-tenant** — Hỗ trợ nhiều cơ quan giám sát
15. **Advanced ML** — Autoencoder, Graph Neural Networks cho relationship-based anomaly detection
16. **Stress Testing Module** — Scenario analysis (what-if) cho macro stress tests

---

## 8. KẾT LUẬN

BankGuard AI 360° là một hệ thống giám sát ngân hàng được thiết kế tốt với kiến trúc hybrid (Rule + ML) mạnh mẽ. Điểm nổi bật nhất là cơ chế consensus scoring đa thuật toán kết hợp 3 phương pháp XAI (SHAP, Permutation Importance, Local Surrogate) phủ cả 3 mô hình ML — phù hợp cho mục đích giám sát của KTNN.

Hệ thống đã có unit test suite (~120 tests), 9 expert rules Basel III, và XAI toàn diện. Để đưa vào production thực tế, cần ưu tiên: (1) database backend, (2) authentication, (3) tách kiến trúc monolithic, và (4) CI/CD pipeline.

**Tổng điểm đánh giá: 8.5/10** — Kiến trúc ML mạnh, XAI toàn diện, có unit tests. Cần cải thiện infrastructure.

---

> *Báo cáo được tạo bởi Augment Agent — Project Analysis Consultant*
> *Ngày: 2026-03-03*