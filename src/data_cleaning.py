"""
Module làm sạch dữ liệu điện thoại.

Các bước thực hiện:
1. Đọc dữ liệu raw
2. Chuẩn hóa tên hãng (viết hoa/thường)
3. Chuẩn hóa đơn vị giá (đưa về triệu VND)
4. Xử lý giá trị thiếu (imputation theo median/mode)
5. Loại bỏ trùng lặp
6. Loại bỏ outliers (dùng IQR)
7. Lưu file dữ liệu sạch
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "brand", "model", "ram_gb", "storage_gb", "battery_mah",
    "camera_mp", "screen_inch", "cpu_cores", "price_million_vnd",
]

COLUMN_ALIASES = {
    "brand": ["company", "brand", "manufacturer"],
    "model": ["name", "model", "phone_name", "model_name", "device_name"],
    "ram_gb": ["ram", "ram_gb", "memory"],
    "storage_gb": ["external_memory", "rom", "storage", "internal_memory", "storage_gb"],
    "battery_mah": ["battery", "battery_power", "battery_mah"],
    "camera_mp": ["camera", "primary_cam", "primary_camera", "rear_camera", "camera_mp"],
    "screen_inch": ["display", "screen_size", "mobile_size", "screen", "screen_inch"],
    "cpu_cores": ["cpu_cores", "cores", "processor_cores", "core_count", "cpu"],
    "price_million_vnd": ["price", "price_million_vnd", "cost"],
}


def parse_first_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "":
        return np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    if not numbers:
        return np.nan
    return float(numbers[0])


def parse_cpu_cores(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    # Detect explicit number first
    numeric = parse_first_number(text)
    if not np.isnan(numeric) and numeric >= 1:
        return numeric

    if "octa" in text:
        return 8.0
    if "hexa" in text:
        return 6.0
    if "quad" in text:
        return 4.0
    if "dual" in text:
        return 2.0
    if "deca" in text:
        return 10.0
    return np.nan


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột input thành cột chuẩn của pipeline."""
    df = df.copy()
    lower_map = {col.lower().strip(): col for col in df.columns}
    rename_map = {}

    for target, candidates in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for alias in candidates:
            if alias.lower() in lower_map:
                rename_map[lower_map[alias.lower()]] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"🔧 Đã chuẩn hóa tên cột: {rename_map}")

    if "cpu_cores" not in df.columns:
        inferable = [col for col in df.columns if col.lower() in {"processor", "processor_name", "chipset", "cpu"}]
        if inferable:
            df["cpu_cores"] = df[inferable[0]].apply(parse_cpu_cores)
            if df["cpu_cores"].isna().all():
                df["cpu_cores"] = np.nan
                print("⚠️ Không thể suy ra cpu_cores từ cột processor/chipset; tạo cột cpu_cores với giá trị thiếu.")
        else:
            df["cpu_cores"] = np.nan
            print("⚠️ Không tìm thấy cột cpu_cores; tạo cột cpu_cores với giá trị thiếu để tự động điền sau.")

    return df


def load_data(path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ CSV."""
    df = pd.read_csv(path)
    df = standardize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV thiếu cột bắt buộc sau khi chuẩn hóa: {missing}. "
            "Hãy đảm bảo file có ít nhất một trong các tên cột tương ứng."
        )
    print(f"📂 Đã đọc {len(df)} dòng từ {path}")
    return df


def parse_camera_mp(value):
    """Chuyển camera_mp dạng '12 + 12 + 12' hoặc '48 + 50 + 8 + 2' thành số."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "":
        return np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    if not numbers:
        return np.nan
    return sum(float(num) for num in numbers)


def normalize_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo các cột numeric có kiểu số trước khi xử lý tiếp."""
    df = df.copy()

    if "ram_gb" in df.columns:
        df["ram_gb"] = df["ram_gb"].apply(parse_first_number)
    if "storage_gb" in df.columns:
        df["storage_gb"] = df["storage_gb"].apply(parse_first_number)
    if "battery_mah" in df.columns:
        df["battery_mah"] = df["battery_mah"].apply(parse_first_number)
    if "screen_inch" in df.columns:
        df["screen_inch"] = df["screen_inch"].apply(parse_first_number)
    if "cpu_cores" in df.columns:
        df["cpu_cores"] = df["cpu_cores"].apply(parse_cpu_cores)
    if "camera_mp" in df.columns:
        df["camera_mp"] = df["camera_mp"].apply(parse_camera_mp)

    numeric_cols = [
        "ram_gb", "storage_gb", "battery_mah", "screen_inch",
        "cpu_cores", "price_million_vnd"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên hãng: viết chữ cái đầu hoa."""
    df = df.copy()
    df["brand"] = df["brand"].str.strip().str.title()
    # Sửa một vài trường hợp đặc biệt
    df["brand"] = df["brand"].replace({
        "Iphone": "Apple",
        "Asus": "Asus",
    })
    print(f"✅ Chuẩn hóa brand: {df['brand'].nunique()} hãng khác nhau")
    return df


def normalize_price_unit(df: pd.DataFrame, price_col: str = "price_million_vnd") -> pd.DataFrame:
    """
    Chuẩn hóa đơn vị giá về triệu VND.
    Nếu giá > 1000 → giả định là VND, chia 1,000,000.
    Nếu giá > 100 nhưng < 1000 → giả định nghìn VND, chia 1000.
    """
    df = df.copy()

    # Giá > 1,000,000 chắc chắn là VND thô
    mask_vnd = df[price_col] > 1_000_000
    n_vnd = mask_vnd.sum()
    df.loc[mask_vnd, price_col] = df.loc[mask_vnd, price_col] / 1_000_000

    # Giá 100,000 - 1,000,000 có thể là nghìn VND (ít gặp)
    mask_thousand = (df[price_col] > 100) & (df[price_col] <= 1_000_000)
    n_thousand = mask_thousand.sum()
    df.loc[mask_thousand, price_col] = df.loc[mask_thousand, price_col] / 1000

    print(f"✅ Chuẩn hóa đơn vị giá: {n_vnd} dòng VND → triệu VND, "
          f"{n_thousand} dòng nghìn VND → triệu VND")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý giá trị thiếu:
    - Cột số: điền bằng median
    - Cột categorical: điền bằng mode (giá trị xuất hiện nhiều nhất)
    - Giá (target): xóa dòng thiếu vì không thể dự đoán nếu thiếu target
    """
    df = df.copy()
    print(f"\n📊 Số giá trị thiếu trước xử lý:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Xóa dòng thiếu giá (target)
    before = len(df)
    df = df.dropna(subset=["price_million_vnd"])
    print(f"🗑  Xóa {before - len(df)} dòng thiếu giá (không thể huấn luyện)")

    # Điền giá trị thiếu cho các cột số bằng median theo hãng
    numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            # Điền theo median của từng hãng, fallback về median toàn bộ
            df[col] = df.groupby("brand")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    # Cột categorical
    if "brand" in df.columns and df["brand"].isnull().any():
        df["brand"] = df["brand"].fillna(df["brand"].mode()[0])

    print(f"✅ Sau xử lý: {df.isnull().sum().sum()} giá trị thiếu còn lại")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ dòng trùng lặp dựa trên tất cả các cột."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"✅ Loại bỏ {before - len(df)} dòng trùng lặp")
    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str = "price_million_vnd",
                        k: float = 3.0) -> pd.DataFrame:
    """
    Loại bỏ outlier dùng IQR.
    k = 1.5 là chuẩn, k = 3.0 chỉ loại outlier cực đoan (giữ lại flagships).
    """
    df = df.copy()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
    print(f"✅ Loại {before - len(df)} outlier (giá ngoài [{lower:.2f}, {upper:.2f}] triệu VND)")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra và lọc giá trị phi thực tế:
    - RAM: 1-32 GB
    - Storage: 16-2048 GB
    - Pin: 1000-10000 mAh
    - Camera: 1-300 MP (parse từ string nếu cần)
    - Screen: 3.5-8.0 inch
    - Giá: 0.5-100 triệu VND
    """
    df = normalize_numeric_fields(df)

    try:
        camera_condition = df["camera_mp"].between(1, 300)
    except (TypeError, ValueError):
        camera_condition = pd.Series([True] * len(df), index=df.index)

    try:
        screen_condition = df["screen_inch"].between(3.5, 8.0)
    except (TypeError, ValueError):
        screen_condition = pd.Series([True] * len(df), index=df.index)

    conditions = (
        df["ram_gb"].between(1, 32) &
        df["storage_gb"].between(16, 2048) &
        df["battery_mah"].between(1000, 10000) &
        camera_condition &
        screen_condition &
        df["price_million_vnd"].between(0.5, 100)
    )
    before = len(df)
    df = df[conditions].reset_index(drop=True)
    print(f"✅ Loại {before - len(df)} dòng có giá trị phi thực tế")
    return df


def clean_pipeline(input_path: str = "data/mobile_phones.csv",
                   output_path: str = "data/mobile_phones_clean.csv") -> pd.DataFrame:
    """Chạy toàn bộ pipeline làm sạch."""
    print("=" * 60)
    print("🧹 BẮT ĐẦU LÀM SẠCH DỮ LIỆU")
    print("=" * 60)

    df = load_data(input_path)
    df = normalize_brand(df)
    df = normalize_price_unit(df)
    df = normalize_numeric_fields(df)
    df = handle_missing_values(df)
    df = validate_ranges(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df)

    # Lưu kết quả
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT - Đã lưu tại: {output.absolute()}")
    print(f"📊 Dữ liệu sạch: {len(df)} dòng, {df['brand'].nunique()} hãng")
    print(f"💰 Khoảng giá: {df['price_million_vnd'].min():.2f} - "
          f"{df['price_million_vnd'].max():.2f} triệu VND")
    print("=" * 60)
    return df


if __name__ == "__main__":
    clean_pipeline()