# config.py

# 1. Thông tin định danh và phân loại
IDENTIFIERS = ['bank_id', 'period', 'bank_type', 'region', 'stability', 'external_credit_rating']

# 2. Nhóm chỉ số Sức khỏe tài chính (Dùng cho ML Anomaly Detection)
FINANCIAL_HEALTH_FEATURES = [
    'capital_adequacy_ratio', 'npl_ratio', 'liquidity_coverage_ratio', 
    'nsfr', 'provision_coverage_ratio', 'loan_to_deposit_ratio',
    'return_on_assets', 'return_on_equity', 'net_interest_margin'
]

# 3. Nhóm chỉ số Tập trung rủi ro (Dùng cho ML Anomaly Detection)
CONCENTRATION_RISK_FEATURES = [
    'sector_concentration_hhi', 'top20_borrower_concentration', 
    'geographic_concentration', 'top20_depositors_ratio', 'top5_depositors_ratio'
]

# 4. Nhóm chỉ số Ngoại bảng & Thanh khoản (Dùng cho ML Anomaly Detection)
EXPOSURE_LIQUIDITY_FEATURES = [
    'derivatives_to_assets_ratio', 'unused_lines_to_loans_ratio', 
    'guarantees_to_loans_ratio', 'obs_to_assets_ratio', 
    'wholesale_dependency_ratio', 'liquidity_concentration_risk'
]

# 5. Nhóm dữ liệu ngành (Dùng để vẽ Heatmap/Stack Bar Chart)
SECTOR_LOANS_COLUMNS = [
    'sector_loans_energy', 'sector_loans_real_estate', 
    'sector_loans_construction', 'sector_loans_services', 'sector_loans_agriculture'
]

# gom tất cả các feature dùng cho Machine Learning
ALL_ML_FEATURES = FINANCIAL_HEALTH_FEATURES + CONCENTRATION_RISK_FEATURES + EXPOSURE_LIQUIDITY_FEATURES

# 6. Cấu hình Model
MODEL_PARAMS = {
    'contamination': 0.05,  # Tỷ lệ bất thường dự kiến
    'n_clusters': 3,        # Số cụm rủi ro
    'random_state': 42
}