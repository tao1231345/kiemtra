"""
Module trực quan hóa dữ liệu điện thoại.

Bao gồm 4 loại biểu đồ chính theo yêu cầu:
1. Biểu đồ cột: giá trung bình theo hãng
2. Scatter plot: giá vs RAM / bộ nhớ
3. Heatmap: tương quan giữa các thuộc tính
4. Histogram: phân bố giá điện thoại

Cộng thêm vài biểu đồ bổ sung: boxplot theo hãng, pairplot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Cấu hình chung cho tất cả biểu đồ
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 11
# Hỗ trợ ký tự tiếng Việt
plt.rcParams["font.family"] = ["DejaVu Sans"]


def plot_avg_price_by_brand(df: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ cột: giá trung bình theo hãng."""
    brand_avg = df.groupby("brand")["price_million_vnd"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", n_colors=len(brand_avg))
    bars = ax.bar(brand_avg.index, brand_avg.values, color=colors, edgecolor="black")

    # Thêm nhãn giá trị lên đầu cột
    for bar, value in zip(bars, brand_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{value:.1f}", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Gia trung binh dien thoai theo hang (trieu VND)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hang dien thoai", fontsize=12)
    ax.set_ylabel("Gia trung binh (trieu VND)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = output_dir / "01_avg_price_by_brand.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_price_vs_specs_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot: giá vs RAM, storage, camera, pin."""
    features = [
        ("ram_gb", "RAM (GB)"),
        ("storage_gb", "Bo nho (GB)"),
        ("camera_mp", "Camera (MP)"),
        ("battery_mah", "Pin (mAh)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (col, label) in enumerate(features):
        ax = axes[idx]
        scatter = ax.scatter(
            df[col], df["price_million_vnd"],
            c=df["price_million_vnd"], cmap="plasma",
            alpha=0.6, edgecolors="w", s=40,
        )
        # Đường xu hướng (linear fit)
        z = np.polyfit(df[col], df["price_million_vnd"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Xu huong")

        ax.set_title(f"Gia vs {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("Gia (trieu VND)")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Gia")

    plt.suptitle("Moi quan he giua cau hinh va gia dien thoai",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    path = output_dir / "02_scatter_price_vs_specs.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap tương quan giữa các thuộc tính số."""
    numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                    "screen_inch", "cpu_cores", "price_million_vnd"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # che nửa trên tam giác
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        square=True, linewidths=1, cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Ma tran tuong quan giua cac thuoc tinh",
                 fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = output_dir / "03_correlation_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_price_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram phân bố giá + KDE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram có KDE
    ax1 = axes[0]
    sns.histplot(df["price_million_vnd"], bins=40, kde=True,
                 color="steelblue", edgecolor="black", ax=ax1)
    ax1.axvline(df["price_million_vnd"].mean(), color="red", linestyle="--",
                linewidth=2, label=f"TB = {df['price_million_vnd'].mean():.2f}")
    ax1.axvline(df["price_million_vnd"].median(), color="green", linestyle="--",
                linewidth=2, label=f"Trung vi = {df['price_million_vnd'].median():.2f}")
    ax1.set_title("Phan bo gia dien thoai", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Gia (trieu VND)")
    ax1.set_ylabel("So luong")
    ax1.legend()

    # Boxplot
    ax2 = axes[1]
    sns.boxplot(data=df, x="brand", y="price_million_vnd",
                hue="brand", palette="Set3", ax=ax2, legend=False)
    ax2.set_title("Boxplot gia theo hang", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Hang")
    ax2.set_ylabel("Gia (trieu VND)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = output_dir / "04_price_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_segment_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ bổ sung: phân tích 3 phân khúc giá."""
    df = df.copy()
    df["segment"] = pd.qcut(
        df["price_million_vnd"],
        q=[0, 0.33, 0.66, 1.0],
        labels=["Gia re", "Tam trung", "Cao cap"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Số lượng theo phân khúc
    ax1 = axes[0]
    segment_counts = df["segment"].value_counts().sort_index()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax1.pie(segment_counts, labels=segment_counts.index, colors=colors,
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "w", "linewidth": 2})
    ax1.set_title("Ty le theo phan khuc", fontsize=13, fontweight="bold")

    # 2. RAM trung bình theo phân khúc
    ax2 = axes[1]
    ram_by_seg = df.groupby("segment", observed=True)[["ram_gb", "storage_gb"]].mean()
    ram_by_seg.plot(kind="bar", ax=ax2, color=["#3498db", "#9b59b6"])
    ax2.set_title("Cau hinh TB theo phan khuc", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Phan khuc")
    ax2.set_ylabel("Gia tri trung binh (GB)")
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(["RAM", "Bo nho"])

    # 3. Số lượng điện thoại của mỗi hãng trong phân khúc cao cấp
    ax3 = axes[2]
    premium = df[df["segment"] == "Cao cap"]["brand"].value_counts().head(6)
    bars = ax3.barh(premium.index, premium.values, color=sns.color_palette("rocket_r", len(premium)))
    ax3.set_title("Top hang phan khuc cao cap", fontsize=13, fontweight="bold")
    ax3.set_xlabel("So luong")
    ax3.invert_yaxis()
    for bar, val in zip(bars, premium.values):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontweight="bold")

    plt.tight_layout()
    path = output_dir / "05_segment_analysis.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def run_visualization(data_path: str = "data/mobile_phones_clean.csv",
                      output_dir: str = "outputs/figures") -> None:
    """Tạo tất cả biểu đồ."""
    print("\n" + "=" * 60)
    print("🎨 VẼ BIỂU ĐỒ TRỰC QUAN HÓA")
    print("=" * 60)

    df = pd.read_csv(data_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_avg_price_by_brand(df, out)
    plot_price_vs_specs_scatter(df, out)
    plot_correlation_heatmap(df, out)
    plot_price_distribution(df, out)
    plot_segment_analysis(df, out)

    print(f"\n🎉 Đã lưu tất cả biểu đồ tại: {out.absolute()}/")


if __name__ == "__main__":
    run_visualization()