"""
Module sinh dữ liệu điện thoại mô phỏng thị trường Việt Nam.

Dữ liệu được tạo có:
- Mối quan hệ thực tế giữa cấu hình và giá (để ML model học được)
- Cố ý thêm giá trị thiếu, trùng lặp, bất thường để pipeline làm sạch có việc làm
- Đơn vị giá lẫn lộn (triệu VND, VND) để kiểm thử chuẩn hóa
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Seed để tái lập được kết quả
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Thông tin hãng: (giá trung bình base - triệu VND, hệ số premium, trọng số xuất hiện)
BRAND_INFO = {
    "Apple":   (25.0, 1.8, 0.15),
    "Samsung": (12.0, 1.3, 0.20),
    "Xiaomi":  (8.0,  1.0, 0.15),
    "Oppo":    (9.0,  1.05, 0.12),
    "Vivo":    (8.5,  1.0, 0.10),
    "Realme":  (6.5,  0.9, 0.08),
    "Google":  (20.0, 1.6, 0.04),
    "Nokia":   (5.5,  0.85, 0.06),
    "Huawei":  (11.0, 1.2, 0.05),
    "Asus":    (14.0, 1.25, 0.05),
}

# Các lựa chọn cấu hình thực tế
RAM_CHOICES = [2, 3, 4, 6, 8, 12, 16]
RAM_WEIGHTS = [0.05, 0.08, 0.15, 0.25, 0.25, 0.15, 0.07]

STORAGE_CHOICES = [32, 64, 128, 256, 512, 1024]
STORAGE_WEIGHTS = [0.05, 0.15, 0.35, 0.30, 0.10, 0.05]

CAMERA_CHOICES = [8, 12, 13, 48, 50, 64, 108, 200]
CPU_CORES_CHOICES = [4, 6, 8]
CPU_CORES_WEIGHTS = [0.10, 0.25, 0.65]


def _generate_model_name(brand: str, idx: int) -> str:
    """Sinh tên model giả lập cho từng hãng."""
    patterns = {
        "Apple":   f"iPhone {np.random.choice([11, 12, 13, 14, 15, 16])} {np.random.choice(['', 'Pro', 'Pro Max', 'Plus', 'Mini'])}".strip(),
        "Samsung": f"Galaxy {np.random.choice(['A', 'S', 'M', 'Note', 'Z Flip', 'Z Fold'])}{np.random.randint(10, 55)}",
        "Xiaomi":  f"Redmi {np.random.choice(['Note ', ''])}{np.random.randint(9, 14)} {np.random.choice(['', 'Pro', 'Pro+', '5G'])}".strip(),
        "Oppo":    f"Reno{np.random.randint(6, 12)} {np.random.choice(['', 'Pro', '5G'])}".strip(),
        "Vivo":    f"V{np.random.randint(20, 30)} {np.random.choice(['', 'Pro', 'e'])}".strip(),
        "Realme":  f"Realme {np.random.randint(8, 12)} {np.random.choice(['', 'Pro', 'i'])}".strip(),
        "Google":  f"Pixel {np.random.randint(6, 9)} {np.random.choice(['', 'Pro', 'a'])}".strip(),
        "Nokia":   f"Nokia {np.random.choice(['G', 'X', 'C'])}{np.random.randint(10, 50)}",
        "Huawei":  f"Nova {np.random.randint(8, 12)} {np.random.choice(['', 'Pro', 'SE'])}".strip(),
        "Asus":    f"ROG Phone {np.random.randint(5, 8)}",
    }
    return f"{patterns.get(brand, brand)} #{idx}"


def _calculate_price(brand: str, ram: float, storage: float,
                     camera: float, battery: float, screen: float,
                     cpu_cores: int) -> float:
    """
    Tính giá điện thoại dựa trên cấu hình.
    Công thức mô phỏng thực tế với một ít nhiễu ngẫu nhiên.
    Giá trả về đơn vị: triệu VND.
    """
    base_price, brand_multiplier, _ = BRAND_INFO[brand]

    # Đóng góp của từng yếu tố
    ram_contribution = ram * 0.4
    storage_contribution = np.log2(storage) * 0.8
    camera_contribution = np.log2(camera + 1) * 0.6
    battery_contribution = (battery - 3000) / 1000 * 0.3
    screen_contribution = (screen - 5.0) * 0.5
    cpu_contribution = (cpu_cores - 4) * 0.4

    total_spec_value = (
        ram_contribution + storage_contribution + camera_contribution +
        battery_contribution + screen_contribution + cpu_contribution
    )

    # Giá cuối = (base + giá trị cấu hình) * hệ số hãng * nhiễu
    noise = np.random.normal(1.0, 0.12)  # nhiễu ±12%
    price = (base_price * 0.3 + total_spec_value * 1.2) * brand_multiplier * noise

    # Đảm bảo giá dương và hợp lý
    return max(1.5, min(price, 60.0))


def generate_dataset(n_samples: int = 1500, output_path: str = "data/mobile_phones.csv") -> pd.DataFrame:
    """
    Sinh bộ dữ liệu điện thoại.

    Args:
        n_samples: Số lượng mẫu cần sinh
        output_path: Đường dẫn file CSV xuất ra

    Returns:
        DataFrame chứa dữ liệu đã sinh
    """
    print(f"🔧 Đang sinh {n_samples} mẫu dữ liệu điện thoại...")

    brands = list(BRAND_INFO.keys())
    brand_probs = [info[2] for info in BRAND_INFO.values()]
    # Chuẩn hóa về tổng = 1
    brand_probs = np.array(brand_probs) / sum(brand_probs)

    data = []
    for i in range(n_samples):
        brand = np.random.choice(brands, p=brand_probs)

        ram = np.random.choice(RAM_CHOICES, p=RAM_WEIGHTS)
        storage = np.random.choice(STORAGE_CHOICES, p=STORAGE_WEIGHTS)
        camera = np.random.choice(CAMERA_CHOICES)
        battery = np.random.randint(3000, 6001)
        screen = round(np.random.uniform(5.0, 6.9), 1)
        cpu_cores = np.random.choice(CPU_CORES_CHOICES, p=CPU_CORES_WEIGHTS)

        price = _calculate_price(brand, ram, storage, camera, battery, screen, cpu_cores)

        data.append({
            "brand": brand,
            "model": _generate_model_name(brand, i),
            "ram_gb": ram,
            "storage_gb": storage,
            "battery_mah": battery,
            "camera_mp": camera,
            "screen_inch": screen,
            "cpu_cores": cpu_cores,
            "price_million_vnd": round(price, 2),
        })

    df = pd.DataFrame(data)

    # ─── Cố ý "làm bẩn" dữ liệu để pipeline clean có việc làm ───

    # 1. Thêm giá trị thiếu (~5% mỗi cột)
    for col in ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "price_million_vnd"]:
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. Thêm dòng trùng lặp (~2%)
    n_duplicates = int(0.02 * len(df))
    dup_idx = np.random.choice(df.index, size=n_duplicates, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 3. Thêm một vài giá trị bất thường (outliers)
    outlier_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_idx, "price_million_vnd"] = df.loc[outlier_idx, "price_million_vnd"] * 10  # giá cao vọt

    # 4. Trộn đơn vị: một số dòng để giá ở VNĐ thay vì triệu VNĐ (x1,000,000)
    unit_mixed_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[unit_mixed_idx, "price_million_vnd"] = df.loc[unit_mixed_idx, "price_million_vnd"] * 1_000_000

    # 5. Viết in hoa/thường lẫn lộn ở cột brand
    case_mix_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[case_mix_idx, "brand"] = df.loc[case_mix_idx, "brand"].str.upper()

    # Xáo trộn thứ tự dòng
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Lưu file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"✅ Đã lưu dữ liệu tại: {output.absolute()}")
    print(f"📊 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
    return df


if __name__ == "__main__":
    # Chạy độc lập module này để sinh dữ liệu
    df = generate_dataset(n_samples=1500)
    print("\n5 dòng đầu tiên:")
    print(df.head())
    print("\nThông tin dữ liệu:")
    print(df.info())