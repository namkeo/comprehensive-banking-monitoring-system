# BankGuard AI 360° — Phân tích Dự án Toàn diện

> **Phiên bản:** 4.0 — Tích hợp EWS & Stress-Testing
> **Ngày cập nhật:** 06/03/2026
> **Đơn vị thụ hưởng:** Kiểm toán Nhà nước (KTNN)
> **Nhóm phát triển:** BankGuard AI Team

---

## Mục lục

1. [Phân tích Tổng quan (Executive Summary)](#1-phân-tích-tổng-quan-executive-summary)
2. [Hệ thống Cảnh báo sớm (EWS)](#2-hệ-thống-cảnh-báo-sớm-early-warning-system--ews)
3. [Mô phỏng Kiểm tra sức chịu đựng (Stress-Testing)](#3-mô-phỏng-kiểm-tra-sức-chịu-đựng-stress-testing)
4. [Động cơ Học máy Đa thuật toán (Ensemble ML Engine)](#4-động-cơ-học-máy-đa-thuật-toán-ensemble-ml-engine)
5. [Phân tích 7 Trụ cột Rủi ro](#5-phân-tích-7-trụ-cột-rủi-ro-7-risk-pillars)
6. [Lợi ích đối với Đơn vị Kiểm toán Nhà nước](#6-lợi-ích-đối-với-đơn-vị-kiểm-toán-nhà-nước)
7. [Phụ lục kỹ thuật](#7-phụ-lục-kỹ-thuật)

---

## 1. Phân tích Tổng quan (Executive Summary)

### 1.1. Triết lý giám sát "360 độ"

BankGuard AI 360° là hệ thống giám sát toàn diện ngân hàng được thiết kế dành riêng cho **Kiểm toán Nhà nước (KTNN)**, vận hành trên triết lý **"Không có góc khuất"** — mọi chiều cạnh rủi ro của tổ chức tín dụng đều được đo lường, theo dõi và cảnh báo thông qua dữ liệu chuỗi thời gian (time-series) đa kỳ.

Hệ thống phân tích **10 ngân hàng × 20 kỳ báo cáo** (quý, từ Q1/2018 đến Q4/2022), tổng cộng **200 quan sát** với **66 chỉ tiêu tài chính** trên mỗi quan sát. Dữ liệu được tổ chức theo cấu trúc panel (bank_id × period), cho phép:

- **Phân tích cắt ngang (cross-sectional):** So sánh các ngân hàng cùng kỳ để phát hiện ngân hàng bất thường.
- **Phân tích dọc (time-series):** Theo dõi xu hướng suy thoái hoặc cải thiện của từng ngân hàng qua các kỳ.
- **Phân tích kết hợp (panel):** Nhận diện các mẫu hình rủi ro hệ thống ảnh hưởng đồng thời nhiều ngân hàng.

### 1.2. Kiến trúc Hybrid: Luật chuyên gia + Trí tuệ nhân tạo

Điểm cốt lõi của BankGuard AI là **Kiến trúc Lai (Hybrid Architecture)** kết hợp hai trụ cột:

| Thành phần | Vai trò | Cơ sở pháp lý / Khoa học |
|---|---|---|
| **Luật Chuyên gia (Expert Rules)** | 9 ngưỡng Basel III tuyệt đối — vi phạm = cảnh báo ngay lập tức | Hiệp ước Basel III, Thông tư NHNN |
| **Học máy Ensemble (ML)** | 3 thuật toán unsupervised × 7 trụ cột — phát hiện bất thường tiềm ẩn | Isolation Forest, LOF, One-Class SVM |

**Tại sao cần kết hợp?**

- **Luật chuyên gia** đảm bảo tính tuân thủ pháp lý: Khi CAR < 8% hoặc LCR < 100%, hệ thống phải cảnh báo ngay — không cần chờ thuật toán ML xác nhận.
- **Học máy** phát hiện các bất thường *chưa có luật*: Một ngân hàng có thể tuân thủ mọi ngưỡng Basel III nhưng vẫn có hành vi bất thường (ví dụ: tăng trưởng tín dụng đột biến kết hợp giảm dự phòng — dấu hiệu che giấu nợ xấu).
- **Kết hợp** tạo ra hệ thống phân loại 3 cấp: **Critical** (ML cao + Rule vi phạm), **Warning** (ML trung bình hoặc Rule vi phạm nhẹ), **Normal**.

### 1.3. Tầm quan trọng đối với công tác kiểm toán nhà nước

Trong bối cảnh hệ thống ngân hàng Việt Nam ngày càng phức tạp với các sản phẩm phái sinh, hoạt động ngoại bảng và liên kết chéo giữa các tổ chức tín dụng, phương pháp kiểm toán truyền thống (dựa trên mẫu chọn và kinh nghiệm chuyên gia) bộc lộ nhiều hạn chế:

- **Phạm vi hẹp:** Chỉ kiểm tra được một số ngân hàng mỗi năm.
- **Phản ứng chậm:** Phát hiện vấn đề sau khi đã xảy ra (hậu kiểm).
- **Thiếu tính hệ thống:** Khó nhận diện rủi ro lây lan giữa các nhóm ngân hàng.

BankGuard AI 360° giải quyết cả ba hạn chế trên bằng cách **giám sát toàn bộ hệ thống ngân hàng, liên tục, tự động, và có khả năng dự báo**.

---

## 2. Hệ thống Cảnh báo sớm (Early Warning System — EWS)

### 2.1. Khái niệm "Vận tốc Rủi ro" (Risk Velocity)

Các chỉ tiêu tài chính tĩnh (static ratios) chỉ cho biết ngân hàng đang ở đâu tại một thời điểm, nhưng không cho biết ngân hàng đang **đi về đâu**. Hệ thống EWS giải quyết vấn đề này bằng khái niệm **Vận tốc Rủi ro** — tốc độ thay đổi của các chỉ tiêu then chốt giữa các kỳ báo cáo.

**Cơ chế tính toán:**

Hệ thống theo dõi 3 chỉ tiêu trọng yếu qua hàm `calculate_risk_trends()`:

| Chỉ tiêu gốc | Cột Delta | Công thức | Ý nghĩa |
|---|---|---|---|
| `npl_ratio` | `npl_delta` | `pct_change()` theo bank_id | Tốc độ tăng nợ xấu |
| `capital_adequacy_ratio` | `car_delta` | `pct_change()` theo bank_id | Tốc độ suy giảm vốn |
| `liquidity_coverage_ratio` | `lcr_delta` | `pct_change()` theo bank_id | Tốc độ suy giảm thanh khoản |

Dữ liệu được sắp xếp theo `bank_id` và `period` (chuyển đổi sang datetime), sau đó tính `pct_change()` theo nhóm ngân hàng. Kỳ đầu tiên (không có dữ liệu trước đó) được gán giá trị delta = 0.

### 2.2. Cơ chế nhận diện "Suy thoái nhanh" (Rapid Deterioration)

Hệ thống tự động gắn cờ **"Rapid Deterioration"** khi phát hiện một trong hai điều kiện:

```
deterioration_flag = "Rapid Deterioration"  nếu:
    ├── npl_delta > 20%    (NPL tăng hơn 20% so với kỳ trước)
    └── car_delta < -15%   (CAR giảm hơn 15% so với kỳ trước)
```

**Ý nghĩa thực tiễn:** Một ngân hàng có CAR = 12% (trên ngưỡng 8%) sẽ không bị Expert Rules cảnh báo. Tuy nhiên, nếu CAR giảm từ 15% xuống 12% trong một quý (car_delta = -20%), EWS sẽ gắn cờ "Rapid Deterioration" — cảnh báo rằng nếu xu hướng tiếp diễn, ngân hàng sẽ vi phạm ngưỡng trong 1-2 quý tới.

### 2.3. Điểm EWS tổng hợp (EWS_Score)

Hàm `calculate_ews_score()` tính điểm tổng hợp **EWS_Score** (0-100) cho mỗi quan sát:

```
EWS_Score = 50% × Current_Component + 50% × Velocity_Component
```

**Current Component (50%):**
- Dựa trên `rule_risk_score` (số lượng luật chuyên gia bị vi phạm).
- Chuẩn hóa: `min(rule_risk_score / 5, 1.0) × 100`.

**Velocity Component (50%):**
- Trung bình cộng của 3 sub-scores:
  - `npl_sub = min(max(npl_delta, 0) / 0.5, 1.0) × 100` — NPL tăng là xấu.
  - `car_sub = min(max(-car_delta, 0) / 0.5, 1.0) × 100` — CAR giảm là xấu.
  - `lcr_sub = min(max(-lcr_delta, 0) / 0.5, 1.0) × 100` — LCR giảm là xấu.
- Mỗi sub-score được cap tại mức thay đổi ±50%.

### 2.4. Ý nghĩa: Chuyển từ "Hậu kiểm" sang "Tiền kiểm"

| Phương pháp truyền thống (Hậu kiểm) | BankGuard AI EWS (Tiền kiểm) |
|---|---|
| Phát hiện vi phạm **sau khi** xảy ra | Cảnh báo **trước khi** vi phạm xảy ra |
| Dựa trên ngưỡng tĩnh (CAR < 8%) | Kết hợp ngưỡng tĩnh + vận tốc thay đổi |
| Không phân biệt mức độ khẩn cấp | Phân loại: Stable vs. Rapid Deterioration |
| Kiểm tra định kỳ (hàng năm) | Giám sát liên tục theo quý |

**Dashboard hiển thị (Tab 6 — "Hệ thống Cảnh báo sớm"):**
- 4 KPI cards: Avg EWS Score, Max EWS Score, Rapid Deterioration count, Stable Banks.
- Risk Velocity Heatmap: Ma trận nhiệt hiển thị NPL Δ%, CAR Δ%, LCR Δ%, EWS Score theo ngân hàng.
- Risk Trajectory: Biểu đồ đường Plotly theo dõi NPL/CAR/LCR qua các kỳ cho ngân hàng được chọn.
- EWS Score Area Chart: Biểu đồ diện tích với ngưỡng cảnh báo tại 50 điểm.

---



## 3. Mô phỏng Kiểm tra sức chịu đựng (Stress-Testing)

### 3.1. Luồng xử lý "What-if" Analysis

Module `models/risk_simulators.py` (322 dòng) cung cấp khả năng **mô phỏng kịch bản khủng hoảng** — cho phép kiểm toán viên đặt câu hỏi: *"Điều gì sẽ xảy ra nếu nợ xấu tăng 50%, tiền gửi bị rút 10%, và tài sản mất giá 15%?"*

**Luồng xử lý 5 bước:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Bước 1: Xác thực đầu vào (Validation)                        │
│  → Kiểm tra 15 cột bắt buộc, kiểm tra phạm vi shock params   │
├─────────────────────────────────────────────────────────────────┤
│  Bước 2: Áp dụng cú sốc (_apply_shocks)                       │
│  → NPL shock → Deposit shock → Asset devaluation               │
├─────────────────────────────────────────────────────────────────┤
│  Bước 3: Tái tính toán tỷ lệ an toàn                          │
│  → CAR, LCR, LDR, npl_ratio được tính lại từ số liệu stressed│
├─────────────────────────────────────────────────────────────────┤
│  Bước 4: Phát hiện vi phạm (_detect_breaches)                  │
│  → So sánh stressed ratios với 9 EXPERT_RULES thresholds       │
├─────────────────────────────────────────────────────────────────┤
│  Bước 5: Tổng hợp kết quả (run_stress_scenario)               │
│  → comparison DataFrame + breaches + summary statistics         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2. Ba loại cú sốc và cơ chế tác động

#### Cú sốc 1: Tăng nợ xấu (NPL Shock)

| Bước | Công thức | Ý nghĩa |
|---|---|---|
| 1 | `NPL_stressed = NPL × npl_shock` | Nợ xấu tăng theo hệ số nhân |
| 2 | `npl_ratio_stressed = NPL_stressed / total_loans` | Tỷ lệ nợ xấu tái tính |
| 3 | `incremental_provision = NPL × (npl_shock - 1)` | Dự phòng bổ sung = 100% NPL gia tăng |
| 4 | `tier1_capital_stressed = tier1 - incremental_provision` | Vốn cấp 1 bị ăn mòn |
| 5 | `CAR_stressed = (tier1_stressed + tier2) / RWA` | CAR tái tính |

*Giả định bảo thủ:* Toàn bộ NPL gia tăng phải được trích lập dự phòng 100%, và dự phòng này trực tiếp giảm vốn cấp 1.

#### Cú sốc 2: Rút tiền hàng loạt (Deposit Shock)

| Bước | Công thức | Ý nghĩa |
|---|---|---|
| 1 | `deposits_stressed = deposits × deposit_shock` | Tiền gửi giảm |
| 2 | `deposit_loss = deposits × (1 - deposit_shock)` | Lượng tiền bị rút |
| 3 | `outflows_stressed = outflows + deposit_loss` | Dòng tiền ra tăng |
| 4 | `LCR_stressed = HQLA / outflows_stressed` | LCR tái tính |
| 5 | `LDR_stressed = total_loans / deposits_stressed` | LDR tái tính |

#### Cú sốc 3: Mất giá tài sản (Asset Devaluation)

| Bước | Công thức | Ý nghĩa |
|---|---|---|
| 1 | `assets_stressed = assets × asset_devaluation` | Tổng tài sản giảm |
| 2 | `RWA_stressed = RWA × asset_devaluation` | Tài sản có rủi ro giảm tương ứng |
| 3 | `CAR_stressed = (tier1 + tier2) / RWA_stressed` | CAR tái tính (kết hợp với NPL shock) |

### 3.3. Phát hiện vi phạm sau Stress (Breach Detection)

Sau khi áp dụng cú sốc, hệ thống so sánh các tỷ lệ stressed với **9 ngưỡng EXPERT_RULES**:

| Mã luật | Chỉ tiêu | Điều kiện vi phạm | Mức độ |
|---|---|---|---|
| `CAR_CRITICAL` | `capital_adequacy_ratio` | < 8% | Critical |
| `LCR_CRITICAL` | `liquidity_coverage_ratio` | < 100% | Critical |
| `NSFR_CRITICAL` | `nsfr` | < 100% | Critical |
| `NPL_WARNING` | `npl_ratio` | > 3% | Warning |
| `LDR_WARNING` | `loan_to_deposit_ratio` | > 100% | Warning |
| `HHI_HIGH_CONCENTRATION` | `sector_concentration_hhi` | > 0.25 | Warning |
| `ROA_WARNING` | `return_on_assets` | < 0% | Warning |
| `WHOLESALE_DEPENDENCY` | `wholesale_dependency_ratio` | > 50% | Warning |
| `TOP20_BORROWER` | `top20_borrower_concentration` | > 25% | Warning |

Kết quả trả về bao gồm:
- **comparison:** DataFrame dạng long-format (bank_id, metric, baseline, stressed, delta, delta_pct) cho 10 chỉ tiêu.
- **breaches:** DataFrame với cột `breach_<RULE_ID>` (boolean) và `n_breaches` (tổng số vi phạm).
- **summary:** Thống kê tổng hợp (avg CAR/LCR/NPL trước và sau, tổng breaches, số ngân hàng có vi phạm mới).

### 3.4. Ý nghĩa: Đánh giá khả năng phòng thủ

Stress-testing trả lời câu hỏi quan trọng nhất trong giám sát vĩ mô: **"Hệ thống ngân hàng có đủ sức chống chịu trước các biến cố cực đoan không?"**

**Dashboard hiển thị (Tab 7 — "Kiểm tra sức chịu đựng"):**
- 3 thanh trượt (sliders): NPL Increase (0-200%), Deposit Outflow (0-50%), Asset Devaluation (0-50%).
- 4 `st.metric` cards với `delta`: Avg CAR, Avg LCR, Avg NPL, Total Breaches — hiển thị rõ tác động.
- Bảng chi tiết CAR: Bank ID | Original CAR | Stressed CAR | Δ% | Status (Pass/Fail vs Basel III 8%).
- Biểu đồ cột so sánh tuân thủ: Baseline vs. Stressed cho CAR ≥ 8%, LCR ≥ 100%, NPL ≤ 3%.

---

## 4. Động cơ Học máy Đa thuật toán (Ensemble ML Engine)

### 4.1. Tại sao cần 3 thuật toán thay vì 1?

Mỗi thuật toán phát hiện bất thường (anomaly detection) có **triết lý khác nhau** về định nghĩa "bất thường". Sử dụng đồng thời 3 thuật toán giúp giảm thiểu sai sót và tăng độ tin cậy:

| Thuật toán | Triết lý | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| **Isolation Forest** | Cô lập: Điểm bất thường cần ít phân tách hơn để tách khỏi tập dữ liệu | Nhanh, hiệu quả với dữ liệu nhiều chiều, 300 cây quyết định | Có thể bỏ sót bất thường cục bộ |
| **Local Outlier Factor (LOF)** | Mật độ: Điểm bất thường nằm trong vùng thưa hơn so với k=20 láng giềng | Phát hiện tốt bất thường cục bộ (local anomalies) | Nhạy cảm với tham số k |
| **One-Class SVM** | Biên giới: Học một ranh giới bao quanh dữ liệu bình thường trong không gian kernel RBF | Mạnh với dữ liệu phi tuyến | Chậm hơn với dữ liệu lớn |

### 4.2. Cơ chế Đồng thuận (Consensus Scoring)

Hệ thống thực hiện **Pillar-by-Pillar Consensus** — mỗi trụ cột rủi ro được đánh giá độc lập bởi cả 3 thuật toán trên không gian đặc trưng riêng:

```
Với mỗi trụ cột (7 trụ cột) × mỗi quan sát:
    IF_prediction  ∈ {-1, 1}    (bất thường / bình thường)
    LOF_prediction ∈ {-1, 1}
    SVM_prediction ∈ {-1, 1}

    n_flags = số lượng thuật toán gắn cờ bất thường (-1)

    Consensus Score = {
        n_flags = 3  →  100 (High Risk)
        n_flags = 2  →   66 (Warning)
        n_flags = 1  →   33 (Monitor)
        n_flags = 0  →    0 (Normal)
    }
```

**Overall ML Risk Score** = Trung bình cộng của 7 Pillar Consensus Scores (0-100).

**Final Hybrid Risk Status** = Kết hợp ML Score + Expert Rule Score:
- **Critical:** ML Score ≥ 60 HOẶC rule_risk_score ≥ 3.
- **Warning:** ML Score ≥ 30 HOẶC rule_risk_score ≥ 1.
- **Normal:** Còn lại.

### 4.3. Giải thích AI (Explainable AI — XAI)

Một trong những thách thức lớn nhất khi áp dụng AI trong kiểm toán nhà nước là **tính minh bạch**. Kiểm toán viên cần hiểu *tại sao* thuật toán đánh dấu một ngân hàng là bất thường. BankGuard AI cung cấp 3 phương pháp XAI:

#### Phương pháp 1: SHAP (SHapley Additive exPlanations)

- **Isolation Forest:** Sử dụng `TreeExplainer` (chính xác, nhanh).
- **LOF & One-Class SVM:** Sử dụng `KernelExplainer` với 10 mẫu nền (background samples).
- **Kết quả:** Ma trận SHAP values (n_samples × 26 features) cho mỗi mô hình — cho biết mỗi đặc trưng đóng góp bao nhiêu vào điểm bất thường.

#### Phương pháp 2: Permutation Feature Importance

- Hoán vị ngẫu nhiên từng đặc trưng và đo mức thay đổi trong dự đoán.
- Áp dụng cho cả 3 mô hình (IF, LOF, SVM).
- **Kết quả:** `importances_mean` và `importances_std` cho 26 features × 3 models.

#### Phương pháp 3: Local Surrogate (LIME-style)

- Tạo nhiễu xung quanh điểm cần giải thích (n_perturbations = 50).
- Huấn luyện mô hình Ridge Regression cục bộ với trọng số kernel Gaussian.
- **Kết quả:** Hệ số hồi quy cho 26 features — cho biết đặc trưng nào ảnh hưởng mạnh nhất đến dự đoán tại điểm cụ thể.

**Dashboard hiển thị (Tab 5 — "XAI & Model Evaluation"):**
- Global Feature Importance (SHAP bar chart) cho 3 mô hình.
- Local Waterfall (SHAP waterfall) cho ngân hàng được chọn.
- Permutation Importance comparison across models.
- Local Surrogate coefficients cross-model comparison.

---

## 5. Phân tích 7 Trụ cột Rủi ro (7 Risk Pillars)

Hệ thống BankGuard AI tổ chức **26 đặc trưng ML** thành **7 trụ cột rủi ro** (Risk Pillars), mỗi trụ cột đại diện cho một chiều cạnh riêng biệt của sức khỏe ngân hàng. Kiến trúc này đảm bảo rằng mỗi chiều rủi ro được đánh giá độc lập trước khi tổng hợp.

### Trụ cột 1: Rủi ro Tín dụng (Credit Risk) — 4 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `npl_ratio` | Tỷ lệ nợ xấu / Tổng dư nợ | Chỉ báo trực tiếp chất lượng tài sản |
| `loan_growth_rate` | Tốc độ tăng trưởng tín dụng YoY | Tăng trưởng quá nhanh → rủi ro bong bóng |
| `provision_coverage_ratio` | Dự phòng / Nợ xấu | Khả năng hấp thụ tổn thất tín dụng |
| `non_performing_loans` | Giá trị tuyệt đối nợ xấu (scaled) | Quy mô rủi ro tín dụng |

**Tầm quan trọng:** Rủi ro tín dụng là nguyên nhân hàng đầu gây đổ vỡ ngân hàng. Trụ cột này phát hiện sớm sự suy giảm chất lượng tài sản — đặc biệt quan trọng trong bối cảnh nợ xấu bất động sản và trái phiếu doanh nghiệp.

### Trụ cột 2: Rủi ro Thanh khoản (Liquidity Risk) — 3 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `liquidity_coverage_ratio` | HQLA / Dòng tiền ra ròng 30 ngày (Basel III LCR) | Khả năng chống chịu stress thanh khoản ngắn hạn |
| `nsfr` | Nguồn vốn ổn định sẵn có / Nguồn vốn ổn định yêu cầu | Cân đối cấu trúc nguồn vốn dài hạn |
| `net_cash_outflows_30d` | Dòng tiền ra ròng 30 ngày (scaled) | Áp lực thanh khoản ngắn hạn |

**Tầm quan trọng:** Khủng hoảng thanh khoản có thể biến một ngân hàng có vốn đầy đủ thành mất khả năng thanh toán trong vài ngày. LCR và NSFR là hai trụ cột thanh khoản của Basel III.

### Trụ cột 3: Rủi ro Tập trung (Concentration Risk) — 3 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `sector_concentration_hhi` | Chỉ số HHI phân bổ tín dụng theo ngành | Mức độ tập trung cho vay vào ít ngành |
| `top20_borrower_concentration` | Dư nợ Top-20 khách hàng / Tổng dư nợ | Rủi ro tên đơn lẻ (single-name risk) |
| `geographic_concentration` | HHI phân bổ địa lý | Mức độ tập trung theo vùng |

**Tầm quan trọng:** Tập trung rủi ro là "kẻ giết người thầm lặng" — ngân hàng có thể đạt mọi chỉ tiêu an toàn nhưng sụp đổ khi một ngành hoặc một nhóm khách hàng lớn gặp khó khăn.

### Trụ cột 4: An toàn Vốn (Capital Adequacy) — 3 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `capital_adequacy_ratio` | (Tier1 + Tier2) / RWA — Basel III CAR | Tấm đệm vốn chống tổn thất |
| `tier1_capital` | Vốn cấp 1 (scaled) | Vốn chất lượng cao nhất |
| `risk_weighted_assets` | Tài sản có rủi ro (scaled) | Mẫu số của CAR |

**Tầm quan trọng:** CAR là chỉ tiêu an toàn vốn quan trọng nhất theo Basel III. Ngưỡng tối thiểu 8% là yêu cầu bắt buộc — vi phạm đồng nghĩa với ngân hàng thiếu vốn.

### Trụ cột 5: Hiệu quả Hoạt động (Earnings & Efficiency) — 5 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `return_on_assets` | ROA — Lợi nhuận / Tổng tài sản | Hiệu quả sử dụng tài sản |
| `return_on_equity` | ROE — Lợi nhuận / Vốn chủ sở hữu | Hiệu quả sử dụng vốn |
| `net_interest_margin` | NIM — Thu nhập lãi ròng / Tài sản sinh lãi | Biên lợi nhuận cốt lõi |
| `operating_expenses` | Chi phí hoạt động (scaled) | Kỷ luật chi phí |
| `operating_income` | Thu nhập hoạt động (scaled) | Năng lực tạo thu nhập |

**Tầm quan trọng:** Ngân hàng thua lỗ kéo dài sẽ ăn mòn vốn. ROA < 0% là dấu hiệu cảnh báo nghiêm trọng — hệ thống gắn cờ `ROA_WARNING` khi phát hiện.

### Trụ cột 6: Rủi ro Ngoại bảng (Off-Balance Sheet) — 4 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `obs_exposure_total` | Tổng cam kết ngoại bảng (scaled) | Quy mô rủi ro ẩn |
| `guarantees_issued` | Bảo lãnh phát hành (scaled) | Nghĩa vụ nợ tiềm tàng |
| `obs_to_assets_ratio` | OBS / Tổng tài sản | Đòn bẩy ngoại bảng |
| `derivatives_to_assets_ratio` | Phái sinh / Tổng tài sản | Rủi ro phái sinh |

**Tầm quan trọng:** Hoạt động ngoại bảng thường bị bỏ qua trong kiểm toán truyền thống nhưng có thể tạo ra tổn thất lớn (bài học từ khủng hoảng 2008). Trụ cột này giám sát các cam kết "ẩn" ngoài bảng cân đối.

### Trụ cột 7: Ổn định Nguồn vốn (Funding Stability) — 4 đặc trưng

| Đặc trưng | Ý nghĩa | Vai trò trong giám sát |
|---|---|---|
| `wholesale_dependency_ratio` | Vốn bán buôn / Tổng nợ phải trả | Phụ thuộc nguồn vốn không ổn định |
| `loan_to_deposit_ratio` | Dư nợ / Tiền gửi | Cân đối cho vay - huy động |
| `top20_depositors_ratio` | Top-20 người gửi / Tổng tiền gửi | Tập trung nguồn vốn |
| `deposit_growth_rate` | Tốc độ tăng trưởng tiền gửi YoY | Xu hướng huy động |

**Tầm quan trọng:** Ngân hàng phụ thuộc quá nhiều vào vốn bán buôn ngắn hạn hoặc một nhóm nhỏ người gửi tiền sẽ dễ bị tổn thương khi thị trường biến động. Ngưỡng `wholesale_dependency_ratio > 50%` được gắn cờ cảnh báo.

---

## 6. Lợi ích đối với Đơn vị Kiểm toán Nhà nước

### 6.1. Tối ưu hóa nguồn lực: Khoanh vùng đúng nhóm ngân hàng "vấn đề"

Với nguồn lực kiểm toán có hạn, việc xác định đúng đối tượng kiểm toán là yếu tố quyết định hiệu quả. BankGuard AI cung cấp:

- **Phân loại 3 cấp tự động (Critical / Warning / Normal):** Kiểm toán viên tập trung 80% nguồn lực vào nhóm Critical, 15% vào Warning, và chỉ giám sát từ xa nhóm Normal.
- **Phân cụm K-Means (3 clusters: Low / Medium / High Risk):** Nhận diện nhóm ngân hàng có hồ sơ rủi ro tương đồng — giúp thiết kế chương trình kiểm toán theo nhóm thay vì từng ngân hàng riêng lẻ.
- **EWS Rapid Deterioration:** Ưu tiên kiểm toán đột xuất các ngân hàng có đà suy giảm nhanh, ngay cả khi chưa vi phạm ngưỡng.
- **Stress-test Fail List:** Danh sách ngân hàng không vượt qua kịch bản stress — ưu tiên kiểm tra khả năng phòng thủ.

### 6.2. Bằng chứng khách quan: Thuật toán toán học minh chứng kết luận kiểm toán

Trong kiểm toán nhà nước, mọi kết luận phải có bằng chứng. BankGuard AI cung cấp:

- **Điểm đồng thuận (Consensus Score):** Không dựa vào một thuật toán duy nhất — kết luận "bất thường" chỉ được đưa ra khi ít nhất 2/3 thuật toán đồng ý (majority vote). Điều này tương đương với nguyên tắc "đa số đồng thuận" trong hội đồng kiểm toán.
- **SHAP Values:** Giải thích chính xác *tại sao* một ngân hàng bị đánh dấu — ví dụ: "NPL Ratio đóng góp +0.35 vào điểm bất thường, trong khi CAR đóng góp +0.22". Đây là bằng chứng toán học có thể trình bày trong báo cáo kiểm toán.
- **Expert Rules trích dẫn Basel III:** Mỗi vi phạm luật đều kèm theo mã luật, ngưỡng, và thông điệp giải thích — có thể trích dẫn trực tiếp trong kết luận kiểm toán.
- **Stress-test Comparison Table:** Bảng so sánh Baseline vs. Stressed với delta % — bằng chứng định lượng về mức độ dễ tổn thương.

### 6.3. Giám sát hệ thống vĩ mô: Nhận diện rủi ro lây lan

BankGuard AI không chỉ giám sát từng ngân hàng riêng lẻ mà còn cung cấp góc nhìn hệ thống:

- **Risk Velocity Heatmap:** Nhìn tổng thể toàn bộ hệ thống ngân hàng trên một ma trận — phát hiện khi nhiều ngân hàng cùng suy giảm đồng thời (dấu hiệu rủi ro hệ thống).
- **Cluster DNA Profiling:** Phân tích "DNA" của từng cụm ngân hàng — nhận diện nhóm ngân hàng có cùng mẫu hình rủi ro (ví dụ: nhóm ngân hàng nhỏ phụ thuộc vốn bán buôn).
- **Sector Concentration Analysis:** Phát hiện khi nhiều ngân hàng cùng tập trung cho vay vào một ngành (ví dụ: bất động sản) — rủi ro lây lan nếu ngành đó gặp khó khăn.
- **Stress-test Aggregate:** Đánh giá toàn hệ thống: "Nếu NPL tăng 100%, bao nhiêu ngân hàng sẽ vi phạm CAR 8%?" — câu hỏi mà chỉ giám sát vĩ mô mới trả lời được.

### 6.4. Tổng hợp giá trị

| Năng lực | Phương pháp truyền thống | BankGuard AI 360° |
|---|---|---|
| Phạm vi giám sát | 2-3 ngân hàng/năm | Toàn bộ hệ thống, liên tục |
| Thời gian phát hiện | Hàng tháng đến hàng năm | Tức thì (mỗi kỳ báo cáo) |
| Tính khách quan | Phụ thuộc kinh nghiệm cá nhân | Thuật toán toán học + đồng thuận 3 mô hình |
| Khả năng dự báo | Không có | EWS Risk Velocity + Stress-testing |
| Giải thích kết quả | Mô tả định tính | SHAP values + Feature Importance định lượng |
| Đánh giá hệ thống | Khó thực hiện | Heatmap + Clustering + Aggregate Stress |

---

## 7. Phụ lục kỹ thuật

### 7.1. Kiến trúc mã nguồn

| Tệp | Dòng code | Vai trò |
|---|---|---|
| `app.py` | 1,830 | Dashboard Streamlit 7 tabs |
| `config.py` | 356 | Cấu hình trung tâm: 7 pillars, 26 features, 9 rules, 3 ML models |
| `models/anomaly_detector.py` | 1,000 | Ensemble ML Engine + XAI (SHAP, PI, Local Surrogate) |
| `models/risk_simulators.py` | 322 | Stress-Testing Engine |
| `utils/data_processor.py` | 591 | Pipeline 7 stages + EWS |
| **Tổng mã nguồn** | **4,099** | |

| Tệp test | Dòng test | Số test cases |
|---|---|---|
| `tests/test_config.py` | 229 | 35 |
| `tests/test_data_processor.py` | 465 | 41 |
| `tests/test_anomaly_detector.py` | 617 | ~56 |
| `tests/test_risk_simulators.py` | 251 | 32 |
| **Tổng test** | **1,562** | **~164** |

### 7.2. Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|---|---|---|
| Ngôn ngữ | Python | 3.14.2 |
| Dashboard | Streamlit | 1.54.0 |
| ML Framework | Scikit-learn | 1.8.0 |
| XAI | SHAP | ≥ 0.46.0 |
| Visualization | Plotly | 6.5.2 |
| Data Processing | Pandas / NumPy | 2.3.3 / 2.4.2 |
| Testing | pytest | 9.0.2 |

### 7.3. Pipeline xử lý dữ liệu (7 stages)

```
CSV Input (200 rows × 66 cols)
    │
    ├── Stage 1: Load (CSV ingestion + validation)
    ├── Stage 2: Impute (Median imputation cho numeric columns)
    ├── Stage 3: Validate (Kiểm tra 26 ML feature columns)
    ├── Stage 4: Engineer (risk_to_profit_ratio, efficiency_ratio)
    ├── Stage 5: Expert Rules (9 Basel III rules → rule_violations, rule_risk_score)
    ├── Stage 6: EWS (npl_delta, car_delta, lcr_delta, deterioration_flag, EWS_Score)
    └── Stage 7: Scale (StandardScaler per pillar → 7 scalers)
    │
    ├── Output 1: df_processed (scaled, ML-ready)
    ├── Output 2: df_original (unscaled, with rules + EWS)
    └── Output 3: scalers (dict of 7 StandardScaler objects)
```

### 7.4. Dashboard 7 Tabs

| Tab | Tên | Nội dung chính |
|---|---|---|
| 1 | Executive Summary | KPI cards, Risk Trajectory line chart, Hybrid Status distribution |
| 2 | Multi-Algo Comparison | IF/LOF/SVM agreement heatmap, model concordance |
| 3 | Risk Pillar Deep Dive | Per-pillar scatter plots, outlier coloring |
| 4 | 360 Bank Profiler | Radar chart, rule violations, funding pie, sector allocation |
| 5 | XAI & Model Evaluation | SHAP importance, waterfall, Permutation Importance, Local Surrogate |
| 6 | Hệ thống Cảnh báo sớm (EWS) | Risk Velocity Heatmap, Risk Trajectory, EWS Score area chart |
| 7 | Kiểm tra sức chịu đựng | Shock sliders, CAR stress table, Compliance bar chart |

---

> **Tài liệu này được biên soạn bởi BankGuard AI Team phục vụ công tác kiểm toán nhà nước.**
> **Mọi thuật toán và ngưỡng đều dựa trên cơ sở khoa học (Basel III, Scikit-learn) và có thể kiểm chứng thông qua mã nguồn mở.**