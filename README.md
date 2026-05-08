# 📱 Mobile Price Analytics Web App

Web app phân tích và dự đoán giá điện thoại với giao diện đẹp, đầy đủ 7 trang, **gộp tất cả vào 1 file `app.py` duy nhất**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## ✨ Tính năng

- 🏠 **Dashboard** – KPI cards, biểu đồ tổng quan, insights nhanh
- 📊 **Khám phá Dữ liệu** – Filter đa chiều, tải CSV, thống kê
- 📈 **Phân tích & Insight** – EDA 5 góc nhìn chuyên sâu
- 🎨 **Biểu đồ Trực quan** – 9 loại chart tương tác (Bar, Scatter, Heatmap, Histogram, Box, Bubble, 3D, Radar, Sunburst)
- 🔮 **Dự đoán Giá** – Form nhập cấu hình hoặc chọn preset, kết quả có khoảng tin cậy
- 🤖 **Hiệu năng Mô hình** – So sánh 5 thuật toán (MAE, RMSE, R²)
- ℹ️ **Giới thiệu** – Tài liệu & hướng dẫn sử dụng

## 📂 Cấu trúc project

```
mobile_price_analysis/
│
├── app.py                          # ⭐ TOÀN BỘ UI 7 TRANG ở 1 file (chạy streamlit run app.py)
├── main.py                         # CLI: chạy toàn bộ pipeline
├── requirements.txt                # Dependencies
├── README.md
│
├── src/                            # 🔧 Backend (giữ nguyên)
│   ├── __init__.py
│   ├── data_generator.py           # Sinh dữ liệu mẫu
│   ├── data_cleaning.py            # Làm sạch dữ liệu
│   ├── eda_analysis.py             # Phân tích EDA (CLI)
│   ├── visualization.py            # Biểu đồ Matplotlib (CLI)
│   ├── model_training.py           # Huấn luyện ML
│   └── ui_utils.py                 # Helpers UI cho Streamlit
│
├── .streamlit/
│   └── config.toml                 # Theme Streamlit
│
├── static/
│   └── style.css                   # Custom CSS
│
├── data/                           # Dữ liệu (tự tạo)
└── outputs/                        # Kết quả ML (tự tạo)
    ├── figures/
    ├── models/
    └── reports/
```

## 🚀 Cài đặt & Chạy

### Bước 1: Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu & train model (chỉ cần 1 lần)

```bash
python main.py
```

Hoặc sử dụng file CSV của bạn:

```bash
python main.py --input-file path/to/your/mobile_data.csv
```

Project hiện đã hỗ trợ tự động đổi tên một số cột phổ biến từ dataset Kaggle như:
`Company` → `brand`, `Name` → `model`, `RAM` → `ram_gb`,
`External_Memory` → `storage_gb`, `Battery` → `battery_mah`,
`Camera` → `camera_mp`, `Display` → `screen_inch`, `Price` → `price_million_vnd`.

Lệnh này sẽ:
- ✅ Sinh 1500 dòng dữ liệu mẫu (nếu không có file input)
- ✅ Làm sạch dữ liệu
- ✅ Huấn luyện 5 mô hình và lưu mô hình tốt nhất

### Bước 3: Khởi động web app

```bash
streamlit run app.py
```

Web app tự động mở tại `http://localhost:8501`. Dùng **sidebar bên trái** để chuyển giữa các trang.

## 🎨 Thiết kế

- **Single-file architecture** – Toàn bộ 7 trang trong `app.py` duy nhất, dễ đọc, dễ sửa
- **Sidebar navigation** – Chuyển trang bằng radio button trong sidebar
- **Backend tách biệt** – Logic xử lý data/ML ở `src/`, UI ở `app.py`
- **Custom theme** – Gradient tím-hồng hiện đại, font Inter
- **Plotly charts** – Tương tác mượt, zoom/pan/hover
- **Responsive** – Hoạt động tốt trên mọi kích thước màn hình

## 🤖 Mô hình Machine Learning

| Mô hình | Loại |
|---------|------|
| Linear Regression | Tuyến tính |
| Ridge Regression | Tuyến tính + L2 |
| Decision Tree | Cây |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |

**Chỉ số đánh giá:** MAE, RMSE, R² + Cross-validation 5-fold

## 🔧 Tùy chỉnh

### Dùng dữ liệu thật của bạn

Chuẩn bị file CSV của bạn với các cột:

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `brand` | string | Hãng |
| `model` | string | Tên model |
| `ram_gb` | float | RAM (GB) |
| `storage_gb` | float | Bộ nhớ (GB) |
| `battery_mah` | float | Pin (mAh) |
| `camera_mp` | float | Camera (MP) |
| `screen_inch` | float | Màn hình (inch) |
| `cpu_cores` | int | Số nhân CPU |
| `price_million_vnd` | float | Giá (triệu VNĐ) |

Sau đó chạy pipeline với file của bạn:
```bash
python main.py --input-file path/to/your/mobile_data.csv
```

Hoặc thay thế file mặc định và chạy:
```bash
cp path/to/your/mobile_data.csv data/mobile_phones.csv
python main.py
```

Rồi restart web app.

### Đổi theme

Sửa `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#ffffff"
```

### Thêm trang mới

Mở `app.py`, thêm option mới vào radio navigation rồi thêm nhánh `elif PAGE == "... Tên trang mới":` tương ứng.

## 🧪 Chạy từng module riêng (CLI)

```bash
python -m src.data_generator      # Sinh dữ liệu
python -m src.data_cleaning       # Làm sạch
python -m src.eda_analysis        # EDA trong terminal
python -m src.visualization       # Vẽ matplotlib PNG
python -m src.model_training      # Train 5 models
```

## 👨‍💻 Tác giả

**OanhKV** – Software Engineer, Hà Nội 🇻🇳

---

Made with ❤️ using Streamlit & scikit-learn • 2026
