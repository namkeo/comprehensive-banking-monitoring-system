Project Title: BankGuard AI - Comprehensive Banking Monitoring System

Tech Stack:

    Language: Python 3.x

    UI Framework: Streamlit (để tạo dashboard nhanh chóng)

    Machine Learning: Scikit-learn (Isolation Forest, KMeans)

    Visualization: Plotly, Seaborn

    Data Handling: Pandas, Numpy

Phần 1: Cấu trúc thư mục (Folder Structure)

Yêu cầu Copilot tạo cấu trúc:
Plaintext

/bank-monitoring-system
|-- app.py (File chạy chính Streamlit)
|-- models/
|   |-- anomaly_detector.py (Logic học máy không giám sát)
|-- utils/
|   |-- data_processor.py (Xử lý dữ liệu thô)
|-- data/
|   |-- time_series_dataset_enriched_v2.csv

Phần 2: Luồng xử lý công việc (Workflows)

    Data Loading: Đọc file CSV, xử lý các giá trị NaN (nếu có), chuẩn hóa dữ liệu bằng StandardScaler.

    Feature Selection: Chọn các cột quan trọng: npl_ratio, capital_adequacy_ratio, liquidity_coverage_ratio, roa, roe.

    Unsupervised Modeling:

        Dùng Isolation Forest để gán điểm bất thường (anomaly_score).

        Dùng K-Means để phân thành 3 cụm sức khỏe tài chính.

    UI/Dashboard:

        Trang 1 (Overview): Biểu đồ tổng quan về CAR và NPL toàn hệ thống.

        Trang 2 (Risk Detection): Danh sách các ngân hàng bị thuật toán gắn thẻ "Anomaly" (Màu đỏ).

        Trang 3 (Peer Analysis): So sánh một ngân hàng cụ thể với trung bình ngành.

Phần 3: Các câu lệnh Prompt mẫu cho Copilot

    Prompt 1 (Xử lý dữ liệu): "Write a Python function in data_processor.py to load the csv file, handle missing values, and scale the features: npl_ratio, capital_adequacy_ratio, and liquidity_coverage_ratio using StandardScaler."

    Prompt 2 (Học máy): "Create an AnomalyDetector class in anomaly_detector.py using Scikit-learn's Isolation Forest. It should take the scaled data and return a dataframe with an 'is_anomaly' column and 'anomaly_score'."

    Prompt 3 (Giao diện): "Using Streamlit, create a dashboard in app.py. Include a sidebar to filter by region and bank_type. Display a Plotly scatter plot showing npl_ratio vs capital_adequacy_ratio where anomalies are colored in red."

    Prompt 4 (Giải thích): "Write a function to explain the anomaly score by showing which feature contributed most to the high score (using SHAP or simple feature importance)."

4. Giao diện dự kiến (UI Design Concept)

    Sidebar: Bộ lọc thời gian (period), Vùng miền (region), Loại ngân hàng.

    Top Metric Cards: Hiển thị Tỷ lệ nợ xấu trung bình hệ thống, Tỷ lệ an toàn vốn trung bình.

    Main Chart: Một biểu đồ 3D (nếu có thể) hoặc 2D Scatter Plot phân cụm các ngân hàng. Các điểm dữ liệu nằm tách biệt khỏi đám đông sẽ được bao quanh bởi vòng tròn đỏ.

    Alert Table: Danh sách: Bank ID | Risk Level | Primary Reason (ví dụ: NPL quá cao).