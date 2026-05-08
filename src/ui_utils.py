"""
Module tiện ích dùng chung cho tất cả các trang của web app.
"""

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_cleaning import normalize_numeric_fields, standardize_columns


REQUIRED_COLUMNS = [
    "brand", "model", "ram_gb", "storage_gb", "battery_mah",
    "camera_mp", "screen_inch", "cpu_cores", "price_million_vnd",
]


# Đường dẫn
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "mobile_phones_clean.csv"
RAW_DATA_PATH = ROOT / "data" / "mobile_phones.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "best_model.pkl"
CSS_PATH = ROOT / "static" / "style.css"
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

# Bảng màu chung cho biểu đồ
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#ec4899",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "neutral": "#64748b",
}

# Bảng màu cho các hãng (cố định để thống nhất giữa các chart)
BRAND_COLORS = {
    "Apple":   "#0f172a",
    "Samsung": "#1428a0",
    "Xiaomi":  "#ff6900",
    "Oppo":    "#1ba84c",
    "Vivo":    "#415fff",
    "Realme":  "#ffcd00",
    "Google":  "#4285f4",
    "Nokia":   "#124191",
    "Huawei":  "#c7000b",
    "Asus":    "#00539b",
}


def load_css():
    """Inject CSS vào app."""
    if CSS_PATH.exists():
        with open(CSS_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_brand():
    """Render logo/brand ở sidebar."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>📱 MobilePrice AI</h2>
            <p>Smart Price Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(icon: str, title: str, subtitle: str):
    """Header chung cho mỗi trang."""
    st.markdown(
        f"""
        <div class="app-header fade-in">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(emoji: str, title: str):
    """Tiêu đề section nhỏ trong trang."""
    st.markdown(
        f"""
        <div class="section-header">
            <span class="emoji">{emoji}</span>
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(icon: str, label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Render một KPI card đẹp."""
    delta_html = ""
    if delta:
        css_class = "kpi-delta" if delta_positive else "kpi-delta negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{css_class}">{arrow} {delta}</div>'

    st.markdown(
        f"""
        <div class="kpi-card fade-in">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_box(content: str, kind: str = "info"):
    """Box thông tin màu sắc."""
    css_class = {
        "info": "info-box",
        "success": "success-box",
        "warning": "warning-box",
    }.get(kind, "info-box")

    st.markdown(
        f'<div class="{css_class}">{content}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """Load dữ liệu đã làm sạch (có cache)."""
    if not DATA_PATH.exists():
        st.error(f"❌ Không tìm thấy dữ liệu sạch tại: {DATA_PATH}")
        st.info("💡 Vui lòng chạy `python main.py` trước để sinh dữ liệu và huấn luyện mô hình.")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load dữ liệu raw (có cache)."""
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load model đã huấn luyện (có cache)."""
    import joblib
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def format_vnd(value: float, unit: str = "triệu") -> str:
    """Format số thành chuỗi giá VND."""
    if unit == "triệu":
        return f"{value:,.2f} triệu"
    elif unit == "đồng":
        return f"{value * 1_000_000:,.0f} đ"
    return f"{value:,.2f}"


def setup_page(title: str, icon: str, layout: str = "wide"):
    """Setup cấu hình chung cho 1 trang."""
    st.set_page_config(
        page_title=f"{title} | MobilePrice AI",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    load_css()
    render_sidebar_brand()


def validate_mobile_df(df: pd.DataFrame):
    """Kiểm tra DataFrame import có đủ các cột cần thiết không.

    Trả về (ok, missing_cols). missing_cols rỗng nếu ok=True.
    """
    df = standardize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def normalize_mobile_df(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột numeric cho dữ liệu upload vào app."""
    df = standardize_columns(df)
    return normalize_numeric_fields(df)


def get_active_df() -> pd.DataFrame:
    """Lấy DataFrame đang được chọn để phân tích.

    Nếu user đã import CSV thì dùng file đó, ngược lại dùng dữ liệu mặc định.
    """
    custom = st.session_state.get("df_custom")
    if custom is not None:
        return custom
    return load_clean_data()


def build_excel_report(df: pd.DataFrame) -> bytes:
    """Sinh file Excel báo cáo tổng hợp với nhiều sheet.

    Sheets:
        1. Tổng quan     - KPIs & thống kê chung
        2. Dữ liệu       - Toàn bộ dữ liệu điện thoại
        3. Theo hãng     - Thống kê chi tiết theo hãng
        4. Tương quan    - Ma trận tương quan giữa các thuộc tính
        5. Phân khúc giá - Thống kê theo 3 phân khúc Giá rẻ / Tầm trung / Cao cấp
        6. Top 10 đắt    - 10 điện thoại có giá cao nhất
        7. Top 10 rẻ     - 10 điện thoại có giá thấp nhất
    """
    numeric_cols = [
        "ram_gb", "storage_gb", "battery_mah",
        "camera_mp", "screen_inch", "cpu_cores",
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        wb = writer.book

        # ─── Định dạng chung ───
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#6366f1", "font_color": "white",
            "align": "center", "valign": "vcenter", "border": 1,
        })
        title_fmt = wb.add_format({
            "bold": True, "font_size": 16, "font_color": "#0f172a",
            "align": "left", "valign": "vcenter",
        })
        money_fmt = wb.add_format({"num_format": "#,##0.00", "border": 1})
        int_fmt = wb.add_format({"num_format": "#,##0", "border": 1})
        text_fmt = wb.add_format({"border": 1})

        # ─── Sheet 1: Tổng quan ───
        ws = wb.add_worksheet("Tổng quan")
        writer.sheets["Tổng quan"] = ws
        ws.set_column("A:A", 32)
        ws.set_column("B:B", 22)
        ws.merge_range("A1:B1", "BÁO CÁO PHÂN TÍCH GIÁ ĐIỆN THOẠI", title_fmt)
        ws.write("A2", f"Ngày xuất: {datetime.now():%d/%m/%Y %H:%M}")

        overview_rows = [
            ("Tổng số điện thoại", len(df)),
            ("Số hãng", df["brand"].nunique()),
            ("Số model khác nhau", df["model"].nunique()),
            ("Giá trung bình (triệu VND)", round(df["price_million_vnd"].mean(), 2)),
            ("Giá trung vị (triệu VND)", round(df["price_million_vnd"].median(), 2)),
            ("Giá thấp nhất (triệu VND)", round(df["price_million_vnd"].min(), 2)),
            ("Giá cao nhất (triệu VND)", round(df["price_million_vnd"].max(), 2)),
            ("Độ lệch chuẩn", round(df["price_million_vnd"].std(), 2)),
            ("RAM trung bình (GB)", round(df["ram_gb"].mean(), 2)),
            ("Bộ nhớ trung bình (GB)", round(df["storage_gb"].mean(), 2)),
            ("Camera trung bình (MP)", round(df["camera_mp"].mean(), 2)),
            ("Pin trung bình (mAh)", round(df["battery_mah"].mean(), 0)),
        ]
        ws.write_row("A4", ["Chỉ số", "Giá trị"], header_fmt)
        for i, (k, v) in enumerate(overview_rows, start=5):
            ws.write(f"A{i}", k, text_fmt)
            fmt = int_fmt if isinstance(v, int) else money_fmt
            ws.write(f"B{i}", v, fmt)

        # ─── Sheet 2: Dữ liệu ───
        df_export = df.copy()
        rename_map = {
            "brand": "Hãng", "model": "Model",
            "ram_gb": "RAM (GB)", "storage_gb": "Bộ nhớ (GB)",
            "battery_mah": "Pin (mAh)", "camera_mp": "Camera (MP)",
            "screen_inch": "Màn hình (inch)", "cpu_cores": "Số nhân CPU",
            "price_million_vnd": "Giá (triệu VND)",
        }
        df_export = df_export.rename(columns=rename_map)
        df_export.to_excel(writer, sheet_name="Dữ liệu", index=False)
        ws2 = writer.sheets["Dữ liệu"]
        for col_idx, col_name in enumerate(df_export.columns):
            ws2.write(0, col_idx, col_name, header_fmt)
            width = max(14, min(28, int(df_export[col_name].astype(str).str.len().max() or 10) + 2))
            ws2.set_column(col_idx, col_idx, width)
        ws2.freeze_panes(1, 0)

        # ─── Sheet 3: Theo hãng ───
        brand_stats = df.groupby("brand").agg(
            so_luong=("price_million_vnd", "count"),
            gia_tb=("price_million_vnd", "mean"),
            gia_median=("price_million_vnd", "median"),
            gia_min=("price_million_vnd", "min"),
            gia_max=("price_million_vnd", "max"),
            ram_tb=("ram_gb", "mean"),
            storage_tb=("storage_gb", "mean"),
            camera_tb=("camera_mp", "mean"),
            pin_tb=("battery_mah", "mean"),
        ).round(2).sort_values("gia_tb", ascending=False).reset_index()
        brand_stats.columns = [
            "Hãng", "Số lượng", "Giá TB (tr)", "Giá median (tr)",
            "Giá min (tr)", "Giá max (tr)", "RAM TB (GB)",
            "Bộ nhớ TB (GB)", "Camera TB (MP)", "Pin TB (mAh)",
        ]
        brand_stats.to_excel(writer, sheet_name="Theo hãng", index=False)
        ws3 = writer.sheets["Theo hãng"]
        for col_idx, col_name in enumerate(brand_stats.columns):
            ws3.write(0, col_idx, col_name, header_fmt)
            ws3.set_column(col_idx, col_idx, 16)
        ws3.freeze_panes(1, 0)

        # ─── Sheet 4: Tương quan ───
        corr = df[numeric_cols + ["price_million_vnd"]].corr().round(3)
        label_map = {
            "ram_gb": "RAM", "storage_gb": "Bộ nhớ", "battery_mah": "Pin",
            "camera_mp": "Camera", "screen_inch": "Màn hình",
            "cpu_cores": "CPU", "price_million_vnd": "Giá",
        }
        corr.index = [label_map[c] for c in corr.index]
        corr.columns = [label_map[c] for c in corr.columns]
        corr.to_excel(writer, sheet_name="Tương quan")
        ws4 = writer.sheets["Tương quan"]
        ws4.set_column(0, len(corr.columns), 14)
        for col_idx, col_name in enumerate(corr.columns, start=1):
            ws4.write(0, col_idx, col_name, header_fmt)

        # ─── Sheet 5: Phân khúc giá ───
        df_seg = df.copy()
        df_seg["segment"] = pd.qcut(
            df_seg["price_million_vnd"],
            q=[0, 0.33, 0.66, 1.0],
            labels=["Giá rẻ", "Tầm trung", "Cao cấp"],
        )
        seg_stats = df_seg.groupby("segment", observed=True).agg(
            so_luong=("price_million_vnd", "count"),
            gia_min=("price_million_vnd", "min"),
            gia_max=("price_million_vnd", "max"),
            gia_tb=("price_million_vnd", "mean"),
            ram_tb=("ram_gb", "mean"),
            storage_tb=("storage_gb", "mean"),
            camera_tb=("camera_mp", "mean"),
        ).round(2).reset_index()
        seg_stats.columns = [
            "Phân khúc", "Số lượng", "Giá min (tr)", "Giá max (tr)",
            "Giá TB (tr)", "RAM TB (GB)", "Bộ nhớ TB (GB)", "Camera TB (MP)",
        ]
        seg_stats.to_excel(writer, sheet_name="Phân khúc giá", index=False)
        ws5 = writer.sheets["Phân khúc giá"]
        for col_idx, col_name in enumerate(seg_stats.columns):
            ws5.write(0, col_idx, col_name, header_fmt)
            ws5.set_column(col_idx, col_idx, 18)

        # ─── Sheet 6 & 7: Top đắt / rẻ ───
        top_cols = ["brand", "model", "ram_gb", "storage_gb",
                    "camera_mp", "battery_mah", "price_million_vnd"]
        top_rename = {
            "brand": "Hãng", "model": "Model",
            "ram_gb": "RAM (GB)", "storage_gb": "Bộ nhớ (GB)",
            "camera_mp": "Camera (MP)", "battery_mah": "Pin (mAh)",
            "price_million_vnd": "Giá (triệu VND)",
        }
        top_expensive = (df.nlargest(10, "price_million_vnd")[top_cols]
                         .rename(columns=top_rename).reset_index(drop=True))
        top_cheap = (df.nsmallest(10, "price_million_vnd")[top_cols]
                     .rename(columns=top_rename).reset_index(drop=True))
        top_expensive.to_excel(writer, sheet_name="Top 10 đắt nhất", index=False)
        top_cheap.to_excel(writer, sheet_name="Top 10 rẻ nhất", index=False)
        for sheet_name in ("Top 10 đắt nhất", "Top 10 rẻ nhất"):
            ws_top = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(top_expensive.columns):
                ws_top.write(0, col_idx, col_name, header_fmt)
                ws_top.set_column(col_idx, col_idx, 18)
            ws_top.freeze_panes(1, 0)

    buffer.seek(0)
    return buffer.getvalue()