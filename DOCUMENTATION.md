# BankGuard AI 360 - Tai lieu Ky thuat He thong Giam sat Toan dien Ngan hang

> **Phien ban:** 2.0  |  **Ngay cap nhat:** 02/03/2026  |  **Tac gia:** BankGuard AI Team - KTNN

---

## Muc luc

1. [Tong quan he thong](#1-tong-quan-he-thong)
2. [Cau truc thu muc](#2-cau-truc-thu-muc)
3. [Luong xu ly du lieu (Data Pipeline)](#3-luong-xu-ly-du-lieu-data-pipeline)
4. [Dong co Hoc may (ML Engine)](#4-dong-co-hoc-may-ml-engine)
5. [Cac nhom rui ro trong yeu (Feature Groups)](#5-cac-nhom-rui-ro-trong-yeu-feature-groups)
6. [He thong Quy tac Chuyen gia (Expert Rules Engine)](#6-he-thong-quy-tac-chuyen-gia-expert-rules-engine)
7. [Giao dien Giam sat Dieu hanh (Dashboard)](#7-giao-dien-giam-sat-dieu-hanh-dashboard)
8. [Phu luc ky thuat](#8-phu-luc-ky-thuat)

---

## 1. Tong quan he thong

### 1.1 Gioi thieu

**BankGuard AI 360** la he thong giam sat rui ro ngan hang toan dien, duoc xay dung danh rieng cho cong tac kiem toan va giam sat an toan he thong tai chinh. He thong ket hop **Hoc may khong giam sat (Unsupervised Machine Learning)** voi **Dong co Quy tac Chuyen gia (Expert Rules Engine)** de tao ra mot khuon kho danh gia rui ro **Hybrid** (ket hop) - vua co kha nang phat hien bat thuong tu dong bang AI, vua dam bao tuan thu cac nguong an toan theo chuan **Basel III**.

### 1.2 Muc tieu

| Muc tieu | Mo ta |
|----------|-------|
| **Phan tich da bien (Multivariate Analysis)** | Su dung 26 chi so tai chinh thuoc 7 tru cot rui ro de phat hien bat thuong da chieu |
| **Dong thuan da mo hinh (Multi-Algorithm Consensus)** | 3 thuat toan ML doc lap (Isolation Forest, LOF, One-Class SVM) bieu quyet theo da so cho tung tru cot |
| **Tich hop quy tac Basel III** | 5 nguong quy dinh tuyet doi (CAR, NPL, LCR, LDR, HHI) duoc danh gia song song voi ML |
| **Giai thich duoc (Explainability - XAI)** | He thong chi ra **nguyen nhan goc re** (anomaly driver) cua moi canh bao |
| **Phan cum rui ro (Risk Clustering)** | K-Means (k=3) phan nhom ngan hang theo muc do rui ro va phan tich "DNA" nganh cho vay |
| **Giam sat truc quan** | Dashboard 4 tab Streamlit voi bieu do tuong tac Plotly |

### 1.3 Kien truc tong the

```
CSV Data (66 columns x 200 obs)
       |
       v
[data_processor.py] -- 6-Stage Pipeline --
  Load -> Impute -> Validate -> Engineer -> Rule Engine -> Scale
       |                                        |
       v                                        v
  df_processed (scaled)               df_original (rule_violations,
       |                               rule_risk_score)
       v
[anomaly_detector.py] -- Multi-Algorithm Consensus --
  3 Models x 7 Pillars = 21 model fits
       |
       v
  Consensus Vote -> Overall_ML_Risk_Score
       |                     +
       |            rule_risk_score
       v                     |
  Final_Hybrid_Risk_Status <-+
  + K-Means Clustering + Anomaly Drivers + OBS Risk
       |
       v
[app.py] -- Streamlit Dashboard (4 Tabs) --
  Executive Summary | Multi-Algo Comparison
  Risk Pillar Deep Dive | 360 Bank Profiler
```

---

## 2. Cau truc thu muc

```
comprehensive-banking-monitoring-system/
|
|-- app.py                          # Dashboard Streamlit chinh (964 dong)
|-- config.py                       # Cau hinh trung tam: pillars, rules, ML models (325 dong)
|-- requirements.txt                # Thu vien Python can thiet
|-- DOCUMENTATION.md                # Tai lieu nay
|-- INSTRUCTIONS.md                 # Huong dan su dung
|
|-- data/
|   +-- time_series_dataset_enriched_v2.csv   # Du lieu dau vao (200 x 66)
|
|-- models/
|   +-- anomaly_detector.py         # Dong co ML da thuat toan (653 dong)
|
+-- utils/
    +-- data_processor.py           # Pipeline xu ly du lieu 6 giai doan (402 dong)
```

### Chi tiet vai tro tung file

| File | Vai tro |
|------|---------|
| **config.py** | **Nguon su that duy nhat (Single Source of Truth)** - Dinh nghia 7 tru cot rui ro (26 dac trung ML), 5 quy tac chuyen gia Basel III, cau hinh 3 mo hinh ML, nhan hien thi, va tham so pipeline |
| **utils/data_processor.py** | Pipeline 6 giai doan: Nap CSV, Impute trung vi, Validate 26 features, Feature Engineering (2 chi so phai sinh), Expert Rule Engine (5 quy tac), StandardScaler theo tung tru cot |
| **models/anomaly_detector.py** | Dong co Consensus da thuat toan: 3 mo hinh x 7 tru cot = 21 lan fit. Bieu quyet da so cho diem consensus (0/33/66/100). Tich hop K-Means, Anomaly Drivers, OBS Risk |
| **app.py** | Dashboard Streamlit 4 tab voi bo loc sidebar (Period, Region, Bank Type, Credit Rating). Truc quan hoa bang Plotly |
| **requirements.txt** | Khai bao phien ban: streamlit==1.54.0, pandas==2.3.3, numpy==2.4.2, plotly==6.5.2, scikit-learn==1.8.0 |

---

## 3. Luong xu ly du lieu (Data Pipeline)

Module `utils/data_processor.py` thuc hien pipeline 6 giai doan tu dong:

### 3.1 Giai doan 1 - Nap du lieu (Load)

- Doc file `data/time_series_dataset_enriched_v2.csv` bang `pandas.read_csv()`.
- Kiem tra file ton tai va DataFrame khong rong.
- **Dau ra:** DataFrame tho 200 dong x 66 cot.

### 3.2 Giai doan 2 - Xu ly gia tri thieu (Impute)

- Ap dung **Median Imputation** cho toan bo cot so (numeric).
- Chon trung vi (median) thay vi trung binh (mean) de giam anh huong cua outlier.
- Ghi log so cot da duoc impute.

### 3.3 Giai doan 3 - Kiem dinh du lieu (Validate)

- Doi chieu 26 cot `ALL_ML_FEATURES` tu `config.py` voi DataFrame.
- Neu thieu bat ky cot ML nao: **nem loi `KeyError`** va dung pipeline.
- Cac cot nganh cho vay (`sector_loans_*`) la tuy chon - chi canh bao neu thieu.

### 3.4 Giai doan 4 - Tao chi so phai sinh (Feature Engineering)

He thong tu dong tao 2 chi so phai sinh:

| Chi so | Cong thuc | Y nghia |
|--------|-----------|---------|
| `risk_to_profit_ratio` | `npl_ratio / return_on_assets` | Ty le rui ro/loi nhuan - cang cao cang nguy hiem. Cap tran tai 1,000,000 |
| `efficiency_ratio` | `operating_expenses / operating_income` | Hieu qua hoat dong - gia tri > 1.0 nghia la chi phi vuot thu nhap. Cap tran tai 10.0 |

### 3.5 Giai doan 5 - Dong co Quy tac Chuyen gia (Expert Rule Engine)

- Danh gia **5 quy tac Basel III** tren du lieu **chua chuan hoa** (unscaled).
- Voi moi quan sat (bank-period), he thong:
  1. Kiem tra tung quy tac (cot, toan tu, nguong).
  2. Luu ID cac quy tac vi pham vao cot `rule_violations` (list).
  3. Dem so vi pham vao cot `rule_risk_score` (int).
- Quan sat khong vi pham nhan gia tri `["Compliant"]`.

### 3.6 Giai doan 6 - Chuan hoa theo tung tru cot (Per-Pillar Scaling)

- Ap dung `sklearn.preprocessing.StandardScaler` **rieng biet** cho tung nhom rui ro (7 nhom).
- Moi nhom co scaler doc lap de tranh **feature dominance** (chi so co phuong sai lon at che chi so nho).

**Cac nhom chuan hoa:**

| Nhom | So features | Cot |
|------|-------------|-----|
| `credit_risk` | 4 | npl_ratio, loan_growth_rate, provision_coverage_ratio, non_performing_loans |
| `liquidity_risk` | 3 | liquidity_coverage_ratio, nsfr, net_cash_outflows_30d |
| `concentration_risk` | 3 | sector_concentration_hhi, top20_borrower_concentration, geographic_concentration |
| `capital_adequacy` | 3 | capital_adequacy_ratio, tier1_capital, risk_weighted_assets |
| `earnings_and_efficiency` | 5 | return_on_assets, return_on_equity, net_interest_margin, operating_expenses, operating_income |
| `off-balance_sheet` | 4 | obs_exposure_total, guarantees_issued, obs_to_assets_ratio, derivatives_to_assets_ratio |
| `funding_stability` | 4 | wholesale_dependency_ratio, loan_to_deposit_ratio, top20_depositors_ratio, deposit_growth_rate |

**Dau ra cuoi cung:** `(df_processed, df_original, scalers)`
- `df_processed` - Du lieu da chuan hoa, san sang cho ML.
- `df_original` - Du lieu goc, da bo sung `rule_violations` va `rule_risk_score`.
- `scalers` - Dict chua 7 doi tuong StandardScaler da fit.

---

## 4. Dong co Hoc may (ML Engine)

Module `models/anomaly_detector.py` trieu khai kien truc **Multi-Algorithm Consensus Scoring** voi 4 thanh phan chinh:

### 4.1 Ba mo hinh phat hien bat thuong (Unsupervised Anomaly Detectors)

#### 4.1.1 Isolation Forest (IF)

| Tham so | Gia tri | Giai thich |
|---------|---------|------------|
| `contamination` | 0.05 | Ty le bat thuong ky vong (5%) |
| `n_estimators` | 300 | So cay quyet dinh trong rung |
| `max_features` | 0.5 | Ty le dac trung duoc chon ngau nhien moi cay |
| `random_state` | 42 | Seed dam bao tai lap ket qua |

**Nguyen ly hoat dong:** Co lap quan sat bang cach chon ngau nhien dac trung va gia tri chia. Diem bat thuong can **it lan chia hon** (duong di ngan hon trong cay) nen de bi co lap. Ket qua: `-1` (bat thuong) hoac `1` (binh thuong).

#### 4.1.2 Local Outlier Factor (LOF)

| Tham so | Gia tri | Giai thich |
|---------|---------|------------|
| `n_neighbors` | 20 | So lang gieng gan nhat |
| `contamination` | 0.05 | Ty le bat thuong ky vong |
| `novelty` | True | Cho phep goi `.predict()` tren du lieu moi |
| `metric` | euclidean | Khoang cach Euclid |

**Nguyen ly hoat dong:** So sanh **mat do cuc bo** cua mot diem voi k lang gieng gan nhat. Diem nam trong vung **thua hon dang ke** so voi lang gieng se bi danh dau la outlier.

#### 4.1.3 One-Class SVM

| Tham so | Gia tri | Giai thich |
|---------|---------|------------|
| `kernel` | rbf | Ham nhan Gaussian (Radial Basis Function) |
| `gamma` | scale | Tu dong tinh theo so features |
| `nu` | 0.05 | Gioi han tren cua ty le loi va support vector |

**Nguyen ly hoat dong:** Hoc mot **ranh gioi quyet dinh** bao quanh du lieu binh thuong trong khong gian kernel. Cac diem nam **ngoai ranh gioi** bi phan loai la bat thuong.

### 4.2 Co che Dong thuan (Consensus Mechanism)

He thong chay **3 mo hinh x 7 tru cot = 21 lan fit** doc lap. Voi **moi tru cot** cua **moi ngan hang**:

| So mo hinh dong y (n_flags) | Diem Consensus | Nhan |
|------------------------------|----------------|------|
| 3/3 mo hinh phat hien bat thuong | **100** | High Risk |
| 2/3 mo hinh phat hien bat thuong | **66** | Warning |
| 1/3 mo hinh phat hien bat thuong | **33** | Monitor |
| 0/3 mo hinh phat hien bat thuong | **0** | Normal |

**Diem ML tong hop:**
```
Overall_ML_Risk_Score = trung binh(7 diem consensus tru cot)
```

### 4.3 Ket hop Hybrid: ML + Quy tac -> Trang thai cuoi cung

He thong ket hop `Overall_ML_Risk_Score` voi `rule_risk_score` (tu Expert Rules Engine):

| Trang thai | Dieu kien |
|------------|-----------|
| **Critical** | ML >= 60 **HOAC** rule_risk_score >= 3 |
| **Warning** | ML >= 30 **HOAC** rule_risk_score >= 1 |
| **Normal** | Khong thoa bat ky dieu kien nao o tren |

**Cot tuong thich nguoc (Backward-compatible):**
- `is_anomaly`: `-1` (Critical), `1` (cac trang thai khac)
- `anomaly_score`: `-(Overall_ML_Risk_Score / 100)` - cang am cang bat thuong

### 4.4 Phan cum rui ro - K-Means + Sector DNA

#### K-Means Clustering (k=3)

- Chay K-Means tren toan bo 26 dac trung da chuan hoa.
- **Gan nhan rui ro tu dong** dua tren chi so hop thanh:
  ```
  risk_composite = mean(npl_ratio) - mean(capital_adequacy_ratio)
  ```
  - Cum co `risk_composite` thap nhat = **Low Risk**
  - Cum co `risk_composite` cao nhat = **High Risk**

#### Sector-Loan DNA Profiling

Moi cum duoc phan tich **"DNA" nganh cho vay** dua tren ty trong 5 nganh:
- `sector_loans_energy` (Nang luong)
- `sector_loans_real_estate` (Bat dong san)
- `sector_loans_construction` (Xay dung)
- `sector_loans_services` (Dich vu)
- `sector_loans_agriculture` (Nong nghiep)

Ket qua DNA cho biet dac tinh phan bo cho vay cua tung nhom rui ro, vi du:
> *"High Real Estate Exposure (35.2%) + Construction (22.1%)"*

### 4.5 Giai thich bat thuong (Explainability - XAI)

He thong xac dinh **nguyen nhan goc re** cho moi ngan hang bi danh dau Critical:

1. Tinh **z-score deviation** cua 26 dac trung so voi trung vi toan he thong.
2. Dac trung co **do lech z-score lon nhat** duoc chon lam `anomaly_driver`.
3. Tu `anomaly_driver`, he thong tra nguoc ve nhom rui ro tuong ung (`anomaly_driver_group`).

Vi du: Neu ngan hang X co `npl_ratio` lech nhieu nhat -> `anomaly_driver = "npl_ratio"`, `anomaly_driver_group = "Credit Risk"`.

### 4.6 Danh gia rui ro Ngoai bang (OBS Risk Contribution)

- Su dung chi so `obs_risk_indicator` (neu co trong du lieu).
- Tinh z-score cua chi so OBS cho cac ngan hang Critical.
- Gan nhan `"High OBS Risk"` neu vuot phan vi thu 75 (P75).

---

## 5. Cac nhom rui ro trong yeu (Feature Groups)

He thong to chuc 26 dac trung ML vao **7 tru cot rui ro** (Risk Pillars), duoc dinh nghia trong `config.py`:

### Tru cot 1 - Rui ro Tin dung (Credit Risk) - 4 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `npl_ratio` | Ty le no xau (NPL/Tong du no) | Chi so chat luong tai san cot loi |
| `loan_growth_rate` | Tang truong tin dung YoY | Tang qua nhanh co the bao hieu ha thap chuan |
| `provision_coverage_ratio` | Du phong/NPL | Kha nang hap thu ton that |
| `non_performing_loans` | Gia tri tuyet doi no xau | Quy mo rui ro tin dung |

### Tru cot 2 - Rui ro Thanh khoan (Liquidity Risk) - 3 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `liquidity_coverage_ratio` | HQLA/Dong tien ra rong 30 ngay (Basel III LCR) | Dem thanh khoan ngan han |
| `nsfr` | Von on dinh san co/Von on dinh bat buoc | On dinh cau truc tai tro |
| `net_cash_outflows_30d` | Dong tien ra rong 30 ngay (stress) | Ap luc thanh khoan |

### Tru cot 3 - Rui ro Tap trung (Concentration Risk) - 3 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `sector_concentration_hhi` | Chi so HHI tap trung nganh | HHI > 0.25 = tap trung cao |
| `top20_borrower_concentration` | Top-20 KH vay/Tong du no | Rui ro ten (name concentration) |
| `geographic_concentration` | HHI dia ly | Tap trung vung mien |

### Tru cot 4 - An toan Von (Capital Adequacy) - 3 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `capital_adequacy_ratio` | (Tier1+Tier2)/RWA | CAR < 8% = vi pham Basel III |
| `tier1_capital` | Von cap 1 cot loi | Kha nang hap thu ton that |
| `risk_weighted_assets` | Tai san co rui ro | Quy mo rui ro |

### Tru cot 5 - Hieu qua Kinh doanh (Earnings & Efficiency) - 5 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `return_on_assets` | Ty suat loi nhuan/Tai san (ROA) | Hieu qua su dung tai san |
| `return_on_equity` | Ty suat loi nhuan/Von (ROE) | Sinh loi cho co dong |
| `net_interest_margin` | Bien lai suat rong (NIM) | Thu nhap lai cot loi |
| `operating_expenses` | Chi phi hoat dong | Ky luat chi phi |
| `operating_income` | Thu nhap hoat dong | Nguon thu chinh |

### Tru cot 6 - Ngoai bang Can doi (Off-Balance Sheet) - 4 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `obs_exposure_total` | Tong rui ro ngoai bang | Quy mo nghia vu tiem tang |
| `guarantees_issued` | Bao lanh da phat hanh | Nghia vu bao lanh |
| `obs_to_assets_ratio` | OBS/Tong tai san | Don bay ngoai bang |
| `derivatives_to_assets_ratio` | Phai sinh/Tong tai san | Rui ro phai sinh |

### Tru cot 7 - On dinh Nguon von (Funding Stability) - 4 features

| Dac trung | Mo ta | Y nghia giam sat |
|-----------|-------|------------------|
| `wholesale_dependency_ratio` | Von ban buon/Tong no phai tra | Phu thuoc von ban buon |
| `loan_to_deposit_ratio` | Du no/Tien gui | LDR > 100% = rui ro tai tro |
| `top20_depositors_ratio` | Top-20 nguoi gui/Tong tien gui | Tap trung nguon tien gui |
| `deposit_growth_rate` | Tang truong tien gui YoY | On dinh co so tien gui |

### Tru cot 8 - Phan bo cho vay theo Nganh (Sector Loans) - 5 columns

> *Chi dung cho truc quan hoa va DNA profiling, khong tham gia vec-to ML.*

| Cot | Nganh |
|-----|-------|
| `sector_loans_energy` | Nang luong |
| `sector_loans_real_estate` | Bat dong san |
| `sector_loans_construction` | Xay dung |
| `sector_loans_services` | Dich vu |
| `sector_loans_agriculture` | Nong nghiep |

---

## 6. He thong Quy tac Chuyen gia (Expert Rules Engine)

He thong ap dung 5 nguong quy dinh theo chuan Basel III, duoc danh gia **truoc khi chuan hoa**:

| ID Quy tac | Chi so | Dieu kien | Nguong | Muc do | Thong bao |
|------------|--------|-----------|--------|--------|-----------|
| `CAR_CRITICAL` | `capital_adequacy_ratio` | < | 0.08 (8%) | **Critical** | CAR duoi 8% - ngan hang thieu von (vi pham Basel III Pillar 1) |
| `NPL_WARNING` | `npl_ratio` | > | 0.03 (3%) | Warning | Ty le no xau vuot 3% - suy giam chat luong tin dung |
| `LCR_CRITICAL` | `liquidity_coverage_ratio` | < | 1.0 (100%) | **Critical** | LCR duoi 100% - thieu HQLA cho kich ban stress 30 ngay |
| `LDR_WARNING` | `loan_to_deposit_ratio` | > | 1.0 (100%) | Warning | LDR vuot 100% - du no vuot co so tien gui |
| `HHI_HIGH_CONCENTRATION` | `sector_concentration_hhi` | > | 0.25 | Warning | HHI nganh vuot 0.25 - tap trung cho vay cao |

**Cach tinh:** Moi quan sat nhan `rule_violations` (danh sach ID vi pham) va `rule_risk_score` (so luong vi pham). Gia tri nay duoc **ket hop voi ML Score** de ra `Final_Hybrid_Risk_Status`.

---

## 7. Giao dien Giam sat Dieu hanh (Dashboard)

File `app.py` xay dung giao dien **BankGuard AI 360 Dashboard** tren nen tang **Streamlit** voi truc quan hoa **Plotly**. Giao dien gom thanh loc ben (Sidebar) va 4 tab chinh.

### 7.1 Sidebar - Bo loc

| Bo loc | Mo ta |
|--------|-------|
| **Period** | Chon mot hoac nhieu ky bao cao |
| **Region** | Loc theo vung mien |
| **Bank Type** | Loc theo loai hinh ngan hang |
| **Credit Rating** | Loc theo xep hang tin nhiem ben ngoai |
| **Run Analysis** | Xoa cache va chay lai toan bo pipeline |

### 7.2 Tab 1 - Executive Summary (Tong quan Dieu hanh)

**6 the KPI:**
- Critical (Hybrid) - So ngan hang trang thai Critical
- Warning (Hybrid) - So ngan hang trang thai Warning
- Total Rule Violations - Tong so vi pham quy tac
- High-Conf ML Anomalies - So ngan hang co ML Score >= 60
- System NPL Ratio - Ty le no xau trung binh toan he thong
- Avg CAR - Ty le an toan von trung binh

**Bieu do:**
- **Hybrid Risk Status Distribution** (Pie chart) - Phan bo trang thai Critical/Warning/Normal
- **Risk Trajectory** (Line + Bar chart) - Xu huong diem ML Risk theo thoi gian, ket hop so ngan hang Critical
- **Risk Cluster Distribution** (Bar chart) - Phan bo Low/Medium/High Risk tu K-Means voi thong tin DNA
- **Anomaly Driver Attribution** (Bar chart) - Nhom rui ro gay ra canh bao cho cac ngan hang Critical

### 7.3 Tab 2 - Multi-Algo Comparison (So sanh Da thuat toan)

**Bieu do:**
- **Per-Pillar Consensus Heatmap** - Ma tran nhiet: hang = ngan hang, cot = 7 tru cot, gia tri = diem consensus (0/33/66/100). Mau sac tu xanh (0) den do (100).
- **Model Agreement Matrix** - Ma tran cho thay su dong y/bat dong y giua IF, LOF, SVM cho tung tru cot: "IF Only", "LOF Only", "SVM Only", "IF+LOF", "IF+SVM", "LOF+SVM", "All 3", "None".
- **Overall ML Risk Score Distribution** (Histogram) - Phan bo diem ML toan he thong voi 2 duong nguong Critical (60) va Warning (30).

### 7.4 Tab 3 - Risk Pillar Deep Dive (Phan tich Sau theo Tru cot)

- **Selectbox** chon 1 trong 7 tru cot rui ro.
- **4 the KPI tru cot:** Diem consensus trung binh, So High Risk (3/3), Warning (2/3), Monitor (1/3).
- **Scatter Plots dong** - Ve cac cap dac trung trong tru cot, to mau theo muc consensus:
  - Do = 3/3 mo hinh dong y (bat thuong)
  - Cam = 2/3 mo hinh
  - Vang = 1/3 mo hinh
  - Xanh = Binh thuong
- **Bang thong ke** dac trung (mean, std, min, max, quartiles).
- **Nut tai CSV** du lieu tru cot da chon.

### 7.5 Tab 4 - 360 Bank Profiler (Ho so Ngan hang 360 do)

- **Selectbox** chon Bank ID cu the.
- **Thanh thong tin:** Bank ID, Region, Bank Type, Hybrid Status, ML Risk Score.
- **Rule Violations Alert Box** (hop canh bao do) - Liet ke cac quy tac bi vi pham.
- **Radar Chart 7 tru cot** - So sanh diem consensus cua ngan hang voi trung binh nhom ngang hang (peer group theo bank_type).
- **Per-Pillar Bar Chart** - Diem consensus theo tung tru cot (mau theo muc do).
- **Bank Details** - Cluster, DNA, Anomaly Driver, Driver Group, OBS Risk, Credit Rating.
- **Funding Structure** (2 Pie charts):
  - Wholesale vs Deposit Funding
  - Sector Loan Allocation (5 nganh)
- **Historical Trend** (Line + Bar) - Xu huong ML Score va Rule Violations theo thoi gian.
- **Nut tai CSV** toan bo bao cao.

---

## 8. Phu luc ky thuat

### 8.1 Cac cot dau ra chinh (Output Columns)

**Per-Pillar (7 tru cot x 5 cot = 35 cot):**

| Mau ten cot | Vi du | Giai thich |
|-------------|-------|------------|
| `<key>_IF` | `credit_risk_IF` | Ket qua Isolation Forest (-1/1) |
| `<key>_LOF` | `credit_risk_LOF` | Ket qua LOF (-1/1) |
| `<key>_SVM` | `credit_risk_SVM` | Ket qua One-Class SVM (-1/1) |
| `<key>_n_flags` | `credit_risk_n_flags` | So mo hinh phat hien bat thuong (0-3) |
| `<key>_consensus` | `credit_risk_consensus` | Diem consensus (0/33/66/100) |

**Trong do `<key>` la:** `credit_risk`, `liquidity_risk`, `concentration_risk`, `capital_adequacy`, `earnings_efficiency`, `obs_exposure`, `funding_stability`.

**Cot tong hop (Aggregate):**

| Cot | Giai thich |
|-----|------------|
| `Overall_ML_Risk_Score` | Trung binh 7 diem consensus (0-100) |
| `Final_Hybrid_Risk_Status` | Critical / Warning / Normal |
| `rule_violations` | Danh sach ID quy tac vi pham |
| `rule_risk_score` | So quy tac vi pham (0-5) |
| `is_anomaly` | -1 (Critical) hoac 1 |
| `anomaly_score` | -(ML_Score/100) |
| `cluster_label` | Low Risk / Medium Risk / High Risk |
| `cluster_dna` | Mo ta DNA nganh cho vay |
| `anomaly_driver` | Dac trung gay bat thuong chinh |
| `anomaly_driver_group` | Tru cot rui ro chua anomaly driver |
| `obs_risk_flag` | High OBS Risk / Normal OBS |
| `obs_risk_zscore` | Z-score chi so OBS |

### 8.2 Yeu cau moi truong

| Thanh phan | Phien ban |
|------------|-----------|
| Python | >= 3.10 |
| Streamlit | 1.54.0 |
| Pandas | 2.3.3 |
| NumPy | 2.4.2 |
| Plotly | 6.5.2 |
| scikit-learn | 1.8.0 |

### 8.3 Cach chay he thong

```bash
# 1. Tao moi truong ao
python -m venv .venv

# 2. Kich hoat
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Cai dat thu vien
pip install -r requirements.txt

# 4. Chay dashboard
streamlit run app.py
```

### 8.4 Luu y quan trong

- **Tat ca bieu do Plotly** su dung tham so `width='stretch'` (Streamlit 1.54.0+).
- **Du lieu duoc cache** bang `@st.cache_data` - nhan **Run Analysis** de lam moi.
- **Pipeline chay tu dong** khi mo dashboard lan dau: Load -> Process -> ML -> Render.
- **Nguong Hybrid** duoc hard-code trong `anomaly_detector.py`: Critical >= 60 (ML) hoac >= 3 (rules), Warning >= 30 (ML) hoac >= 1 (rules).

---

> **BankGuard AI 360** - He thong Giam sat Toan dien Ngan hang | KTNN 2026
