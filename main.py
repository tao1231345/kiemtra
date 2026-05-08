"""
Entry point chính chạy toàn bộ pipeline phân tích giá điện thoại.

Luồng xử lý:
    1. Sinh dữ liệu (nếu chưa có)
    2. Làm sạch dữ liệu
    3. Phân tích EDA
    4. Trực quan hóa
    5. Huấn luyện & đánh giá mô hình

Cách chạy:
    python main.py
    python main.py --input-file path/to/your/data.csv
"""

import argparse
from pathlib import Path

from src.data_generator import generate_dataset
from src.data_cleaning import clean_pipeline
from src.eda_analysis import run_eda
from src.visualization import run_visualization
from src.model_training import run_training


def main(input_file="data/mobile_phones.csv"):
    print("\n" + "🚀" * 30)
    print(" " * 20 + "MOBILE PRICE ANALYSIS")
    print("🚀" * 30)

    # Đường dẫn
    raw_data = Path(input_file)
    clean_data = Path("data/mobile_phones_clean.csv")

    # ─── Bước 1: Sinh dữ liệu (bỏ qua nếu đã có) ───
    print("\n━━━━━━━━━━ BƯỚC 1: CHUẨN BỊ DỮ LIỆU ━━━━━━━━━━")
    if not raw_data.exists():
        generate_dataset(n_samples=1500, output_path=str(raw_data))
    else:
        print(f"✅ Đã có dữ liệu tại: {raw_data}")

    # ─── Bước 2: Làm sạch ───
    print("\n━━━━━━━━━━ BƯỚC 2: LÀM SẠCH DỮ LIỆU ━━━━━━━━━━")
    clean_pipeline(str(raw_data), str(clean_data))

    # ─── Bước 3: EDA ───
    print("\n━━━━━━━━━━ BƯỚC 3: PHÂN TÍCH EDA ━━━━━━━━━━")
    run_eda(str(clean_data))

    # ─── Bước 4: Trực quan hóa ───
    print("\n━━━━━━━━━━ BƯỚC 4: TRỰC QUAN HÓA ━━━━━━━━━━")
    run_visualization(str(clean_data))

    # ─── Bước 5: Huấn luyện mô hình ───
    print("\n━━━━━━━━━━ BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ━━━━━━━━━━")
    run_training(str(clean_data))

    # ─── Kết thúc ───
    print("\n" + "🎉" * 30)
    print(" " * 22 + "HOÀN TẤT TOÀN BỘ PIPELINE!")
    print("🎉" * 30)
    print("\n📁 Kết quả xuất tại thư mục:")
    print("   - outputs/figures/    → Biểu đồ PNG")
    print("   - outputs/models/     → Mô hình đã huấn luyện (.pkl)")
    print("   - outputs/reports/    → Báo cáo CSV")
    print("\n💡 Để dự đoán điện thoại mới:")
    print("   from src.model_training import predict_new_phone")
    print("   price = predict_new_phone('outputs/models/best_model.pkl', {...})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mobile price analysis pipeline")
    parser.add_argument("--input-file", default="data/mobile_phones.csv", help="Path to input CSV file")
    args = parser.parse_args()
    main(args.input_file)