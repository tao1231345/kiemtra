"""
Module phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis).

Thực hiện:
1. Thống kê mô tả
2. Phân tích giá theo hãng
3. Phân tích tương quan cấu hình - giá
4. Phân khúc giá (rẻ / tầm trung / cao cấp)
5. Insight đặc điểm điện thoại giá cao
"""

import pandas as pd
from pathlib import Path


def descriptive_stats(df: pd.DataFrame) -> None:
    """In thống kê mô tả tổng quan."""
    print("\n" + "=" * 60)
    print("📊 1. THỐNG KÊ MÔ TẢ TỔNG QUAN")
    print("=" * 60)

    print(f"\n📐 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
    print(f"🏷  Số hãng: {df['brand'].nunique()}")
    print(f"📱 Số model: {df['model'].nunique()}")

    print("\n📈 Thống kê các biến số:")
    numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                    "screen_inch", "cpu_cores", "price_million_vnd"]
    print(df[numeric_cols].describe().round(2))


def analyze_price_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích giá theo hãng."""
    print("\n" + "=" * 60)
    print("🏷  2. PHÂN TÍCH GIÁ THEO HÃNG")
    print("=" * 60)

    brand_stats = df.groupby("brand").agg(
        so_luong=("price_million_vnd", "count"),
        gia_tb=("price_million_vnd", "mean"),
        gia_trung_vi=("price_million_vnd", "median"),
        gia_min=("price_million_vnd", "min"),
        gia_max=("price_million_vnd", "max"),
        do_lech_chuan=("price_million_vnd", "std"),
    ).round(2).sort_values("gia_tb", ascending=False)

    print("\n💰 Giá trung bình theo hãng (đã sắp xếp):")
    print(brand_stats.to_string())

    print(f"\n🥇 Hãng có giá trung bình cao nhất: {brand_stats.index[0]} "
          f"({brand_stats.iloc[0]['gia_tb']:.2f} triệu VND)")
    print(f"🥉 Hãng có giá trung bình thấp nhất: {brand_stats.index[-1]} "
          f"({brand_stats.iloc[-1]['gia_tb']:.2f} triệu VND)")

    return brand_stats


def analyze_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích tương quan giữa cấu hình và giá."""
    print("\n" + "=" * 60)
    print("🔗 3. TƯƠNG QUAN GIỮA CẤU HÌNH VÀ GIÁ")
    print("=" * 60)

    numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                    "screen_inch", "cpu_cores", "price_million_vnd"]
    corr = df[numeric_cols].corr()

    # Tương quan với giá
    price_corr = corr["price_million_vnd"].drop("price_million_vnd").sort_values(ascending=False)
    print("\n🎯 Tương quan Pearson với giá (càng gần 1 càng ảnh hưởng mạnh):")
    for feature, value in price_corr.items():
        bar = "█" * int(abs(value) * 30)
        direction = "↑" if value > 0 else "↓"
        print(f"   {feature:15s} {direction} {value:+.3f}  {bar}")

    strongest = price_corr.abs().idxmax()
    print(f"\n⭐ Yếu tố ảnh hưởng mạnh nhất: {strongest} "
          f"(hệ số {price_corr[strongest]:+.3f})")

    return corr


def analyze_price_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Phân tích 3 phân khúc giá: rẻ, tầm trung, cao cấp."""
    print("\n" + "=" * 60)
    print("🎚  4. PHÂN KHÚC GIÁ")
    print("=" * 60)

    # Chia dựa trên quantile
    df = df.copy()
    df["segment"] = pd.qcut(
        df["price_million_vnd"],
        q=[0, 0.33, 0.66, 1.0],
        labels=["Giá rẻ", "Tầm trung", "Cao cấp"],
    )

    segment_stats = df.groupby("segment", observed=True).agg(
        so_luong=("price_million_vnd", "count"),
        gia_min=("price_million_vnd", "min"),
        gia_max=("price_million_vnd", "max"),
        gia_tb=("price_million_vnd", "mean"),
        ram_tb=("ram_gb", "mean"),
        storage_tb=("storage_gb", "mean"),
        camera_tb=("camera_mp", "mean"),
        pin_tb=("battery_mah", "mean"),
    ).round(2)

    print("\n📊 Thống kê theo phân khúc:")
    print(segment_stats.to_string())

    print("\n🏭 Hãng chiếm ưu thế ở mỗi phân khúc:")
    for seg in ["Giá rẻ", "Tầm trung", "Cao cấp"]:
        top_brand = df[df["segment"] == seg]["brand"].value_counts().head(3)
        print(f"   {seg:12s}: {', '.join(f'{b} ({c})' for b, c in top_brand.items())}")

    return segment_stats


def find_premium_characteristics(df: pd.DataFrame) -> None:
    """Tìm đặc điểm chung của điện thoại giá cao."""
    print("\n" + "=" * 60)
    print("👑 5. ĐẶC ĐIỂM ĐIỆN THOẠI GIÁ CAO (Top 10%)")
    print("=" * 60)

    threshold = df["price_million_vnd"].quantile(0.9)
    premium = df[df["price_million_vnd"] >= threshold]
    regular = df[df["price_million_vnd"] < threshold]

    print(f"\n💎 Ngưỡng top 10%: ≥ {threshold:.2f} triệu VND "
          f"({len(premium)} điện thoại)")

    print("\n📏 So sánh trung bình (Top 10% vs. Phần còn lại):")
    features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch"]
    for f in features:
        p_mean = premium[f].mean()
        r_mean = regular[f].mean()
        diff_pct = (p_mean - r_mean) / r_mean * 100
        print(f"   {f:15s}: Premium {p_mean:7.2f} | Regular {r_mean:7.2f} | "
              f"Chênh {diff_pct:+.1f}%")

    print("\n🏆 Top 5 hãng thống trị phân khúc cao cấp:")
    top_brands = premium["brand"].value_counts().head(5)
    for brand, count in top_brands.items():
        pct = count / len(premium) * 100
        print(f"   {brand:10s}: {count:3d} điện thoại ({pct:.1f}%)")

    print("\n💡 INSIGHT:")
    print("   - Điện thoại giá cao có xu hướng RAM, bộ nhớ lớn hơn đáng kể")
    print("   - Camera và màn hình cũng là yếu tố quyết định ở phân khúc cao")
    print("   - Các hãng Apple/Samsung/Google chiếm đa số ở phân khúc premium")


def run_eda(data_path: str = "data/mobile_phones_clean.csv",
            report_path: str = "outputs/reports/eda_summary.csv") -> dict:
    """Chạy toàn bộ phân tích EDA và trả về dict kết quả."""
    print("\n" + "🚀" * 30)
    print("BẮT ĐẦU PHÂN TÍCH EDA")
    print("🚀" * 30)

    df = pd.read_csv(data_path)

    descriptive_stats(df)
    brand_stats = analyze_price_by_brand(df)
    corr = analyze_correlation(df)
    segment_stats = analyze_price_segments(df)
    find_premium_characteristics(df)

    # Xuất báo cáo
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    brand_stats.to_csv(report, encoding="utf-8-sig")
    segment_stats.to_csv(
        report.parent / "segment_summary.csv",
        encoding="utf-8-sig",
    )
    corr.to_csv(report.parent / "correlation_matrix.csv", encoding="utf-8-sig")
    print(f"\n📝 Đã xuất báo cáo EDA tại: {report.parent.absolute()}/")

    return {
        "brand_stats": brand_stats,
        "correlation": corr,
        "segment_stats": segment_stats,
    }


if __name__ == "__main__":
    run_eda()