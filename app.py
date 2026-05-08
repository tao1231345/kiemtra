"""
═══════════════════════════════════════════════════════════════
📱 MOBILE PRICE ANALYTICS - WEB APP (Single File)
═══════════════════════════════════════════════════════════════

Web app phân tích và dự đoán giá điện thoại với đầy đủ 7 trang:
    🏠 Dashboard
    📊 Khám phá Dữ liệu
    📈 Phân tích & Insight
    🎨 Biểu đồ Trực quan
    🔮 Dự đoán Giá
    🤖 Hiệu năng Mô hình
    ℹ️  Giới thiệu

Chạy bằng lệnh:
    streamlit run app.py

Backend logic vẫn ở trong thư mục `src/`. File này chỉ chứa UI.
═══════════════════════════════════════════════════════════════
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

from src.ui_utils import (
    BRAND_COLORS,
    COLORS,
    REPORTS_DIR,
    build_excel_report,
    get_active_df,
    load_clean_data,
    load_model,
    load_raw_data,
    render_info_box,
    render_kpi_card,
    render_page_header,
    render_section_header,
    setup_page,
    validate_mobile_df,
)


# ═══════════════════════════════════════════════════════════════
# SETUP: chạy một lần khi app khởi động
# ═══════════════════════════════════════════════════════════════
setup_page(title="Mobile Price Analytics", icon="📱")


def build_feature_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng mức độ ảnh hưởng của từng nhóm đặc trưng đến giá."""
    label_map = {
        "ram_gb": "RAM",
        "storage_gb": "Bộ nhớ",
        "battery_mah": "Pin",
        "camera_mp": "Camera",
        "screen_inch": "Màn hình",
        "cpu_cores": "CPU",
        "brand": "Hãng",
    }
    num_features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores"]
    estimator = model.named_steps.get("model", None)

    if estimator is not None and hasattr(estimator, "feature_importances_"):
        pre = model.named_steps["pre"]
        cat_features = pre.named_transformers_["cat"].get_feature_names_out(["brand"]).tolist()
        imp_df = pd.DataFrame({
            "feature": num_features + cat_features,
            "importance": estimator.feature_importances_,
        })
        imp_df["group"] = imp_df["feature"].apply(lambda x: "brand" if x.startswith("brand_") else x)
        result = imp_df.groupby("group", as_index=False)["importance"].sum()
        result["method"] = "Tầm quan trọng đặc trưng từ mô hình cây"
    elif estimator is not None and hasattr(estimator, "coef_"):
        pre = model.named_steps["pre"]
        cat_features = pre.named_transformers_["cat"].get_feature_names_out(["brand"]).tolist()
        coef_df = pd.DataFrame({
            "feature": num_features + cat_features,
            "importance": np.abs(estimator.coef_),
        })
        coef_df["group"] = coef_df["feature"].apply(lambda x: "brand" if x.startswith("brand_") else x)
        result = coef_df.groupby("group", as_index=False)["importance"].sum()
        result["method"] = "Độ lớn hệ số hồi quy"
    else:
        corr = (
            df[num_features + ["price_million_vnd"]]
            .corr()["price_million_vnd"]
            .drop("price_million_vnd")
            .abs()
            .reset_index()
        )
        corr.columns = ["group", "importance"]
        result = corr
        result["method"] = "Tương quan tuyệt đối với giá"

    result["feature"] = result["group"].map(label_map).fillna(result["group"])
    return result.sort_values("importance", ascending=True)


def get_prediction_sample(model, df: pd.DataFrame):
    """Tạo tập test cố định để vẽ giá thực tế so với giá dự đoán."""
    features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores", "brand"]
    X = df[features]
    y = df["price_million_vnd"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    return X_test, y_test, y_pred, y_test - y_pred


def add_linear_trendline(fig, data: pd.DataFrame, x_col: str, y_col: str = "price_million_vnd") -> None:
    """Draw a linear trendline without requiring statsmodels."""
    trend_data = data[[x_col, y_col]].dropna().sort_values(x_col)
    if len(trend_data) < 2 or trend_data[x_col].nunique() < 2:
        return

    x_values = trend_data[x_col].to_numpy(dtype=float)
    y_values = trend_data[y_col].to_numpy(dtype=float)
    coef = np.polyfit(x_values, y_values, 1)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    y_line = coef[0] * x_line + coef[1]

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="Duong xu huong",
        line=dict(color=COLORS["danger"], width=3),
        hovertemplate="Xu hướng<br>%{x:.2f}, %{y:.2f} tr<extra></extra>",
    ))


FEATURE_LABELS = {
    "price_million_vnd": "Giá",
    "ram_gb": "RAM",
    "storage_gb": "Bộ nhớ",
    "battery_mah": "Pin",
    "camera_mp": "Camera",
    "screen_inch": "Màn hình",
    "cpu_cores": "CPU",
    "brand": "Hãng",
}


def vn_feature_label(feature: str) -> str:
    """Tên tiếng Việt cho các cột kỹ thuật thường dùng trên biểu đồ."""
    return FEATURE_LABELS.get(feature, feature)


def render_chart_comment(content: str) -> None:
    """Hiển thị nhận xét ngắn ngay dưới từng biểu đồ."""
    render_info_box(f"💬 <b>Nhận xét:</b> {content}", kind="info")


def corr_strength(value: float) -> str:
    value = abs(value)
    if value >= 0.7:
        return "mạnh"
    if value >= 0.4:
        return "trung bình"
    if value >= 0.2:
        return "yếu"
    return "rất yếu"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — ĐIỀU HƯỚNG + IMPORT CSV + EXPORT EXCEL
# ═══════════════════════════════════════════════════════════════
if "df_custom" not in st.session_state:
    st.session_state.df_custom = None
    st.session_state.df_custom_name = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

with st.sidebar:
    PAGE = option_menu(
        menu_title=None,
        options=[
            "🏠 Dashboard",
            "📊 Khám phá Dữ liệu",
            "📈 Phân tích & Insight",
            "🎨 Biểu đồ Trực quan",
            "🔮 Dự đoán Giá",
            "🤖 Hiệu năng Mô hình",
            "ℹ️  Giới thiệu",
        ],
        icons=[""] * 7,  # emoji có sẵn trong label
        default_index=0,
        styles={
            "container": {
                "padding": "0.25rem 0.25rem 0.5rem",
                "background": "transparent",
            },
            "nav-link": {
                "font-size": "0.95rem",
                "font-weight": "600",
                "color": "#475569",
                "text-align": "left",
                "margin": "0.15rem 0",
                "padding": "0.7rem 0.9rem",
                "border-radius": "10px",
                "--hover-color": "#eef2ff",
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                "color": "white",
                "font-weight": "700",
                "box-shadow": "0 4px 10px rgba(99, 102, 241, 0.25)",
            },
        },
    )

    # ─── IMPORT CSV ───
    with st.expander("📥 Nhập dữ liệu CSV", expanded=False):
        uploaded = st.file_uploader(
            "Kéo thả hoặc chọn file CSV",
            type=["csv"],
            label_visibility="collapsed",
            key="csv_uploader",
        )
        if uploaded is not None and st.session_state.df_custom_name != uploaded.name:
            try:
                df_up = pd.read_csv(uploaded)
                ok, missing = validate_mobile_df(df_up)
                if not ok:
                    st.error(
                        "❌ CSV thiếu cột bắt buộc:\n\n- "
                        + "\n- ".join(missing)
                    )
                else:
                    st.session_state.df_custom = df_up
                    st.session_state.df_custom_name = uploaded.name
                    st.success(f"✅ Đã tải {len(df_up):,} dòng từ `{uploaded.name}`")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi đọc file: {e}")

        if st.session_state.df_custom is not None:
            st.caption(f"📄 Đang dùng: **{st.session_state.df_custom_name}** "
                       f"({len(st.session_state.df_custom):,} dòng)")
            if st.button("🔄 Quay về dữ liệu mặc định", use_container_width=True):
                st.session_state.df_custom = None
                st.session_state.df_custom_name = None
                st.rerun()
        else:
            st.caption("💡 Cột yêu cầu: brand, model, ram_gb, storage_gb, "
                       "battery_mah, camera_mp, screen_inch, cpu_cores, "
                       "price_million_vnd")

    # Load dữ liệu ACTIVE (custom hoặc default)
    df = get_active_df()

    # ─── EXPORT EXCEL ───
    with st.expander("📤 Xuất báo cáo Excel", expanded=False):
        st.caption("Báo cáo gồm 7 sheet: tổng quan, dữ liệu, theo hãng, "
                   "tương quan, phân khúc, top đắt/rẻ.")
        excel_bytes = build_excel_report(df)
        fname = f"bao_cao_dien_thoai_{datetime.now():%Y%m%d_%H%M}.xlsx"
        st.download_button(
            label="⬇️ Tải báo cáo Excel",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary",
        )

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#                          1) DASHBOARD
# ═══════════════════════════════════════════════════════════════
if PAGE == "🏠 Dashboard":
    st.sidebar.info("Trang tổng quan hiển thị các chỉ số chính và insight nhanh.")

    render_page_header(
        icon="📱",
        title="Mobile Price Analytics",
        subtitle="Phân tích thị trường điện thoại & Dự đoán giá bằng Machine Learning",
    )

    # ─── KPI Cards ───
    render_section_header("📊", "Tổng quan thị trường")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("📱", "Tổng số điện thoại", f"{len(df):,}",
                        delta=f"{df['model'].nunique()} models")
    with c2:
        render_kpi_card("🏷️", "Số hãng", f"{df['brand'].nunique()}", delta="Thương hiệu")
    with c3:
        render_kpi_card("💰", "Giá trung bình", f"{df['price_million_vnd'].mean():.1f}tr",
                        delta=f"Median: {df['price_million_vnd'].median():.1f}tr")
    with c4:
        render_kpi_card("📈", "Khoảng giá",
                        f"{df['price_million_vnd'].min():.1f} - {df['price_million_vnd'].max():.1f}",
                        delta="triệu VNĐ")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Quick Stats ───
    render_section_header("⚡", "Số liệu nhanh")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        row = df.loc[df["price_million_vnd"].idxmax()]
        st.metric("💎 Đắt nhất", f"{row['price_million_vnd']:.2f} tr",
                  delta=row["brand"], delta_color="off")
    with c2:
        row = df.loc[df["price_million_vnd"].idxmin()]
        st.metric("💵 Rẻ nhất", f"{row['price_million_vnd']:.2f} tr",
                  delta=row["brand"], delta_color="off")
    with c3:
        top_brand = df["brand"].value_counts().idxmax()
        top_count = df["brand"].value_counts().max()
        st.metric("🏆 Hãng phổ biến", top_brand, delta=f"{top_count} máy", delta_color="off")
    with c4:
        st.metric("⚙️ Cấu hình TB", f"{df['ram_gb'].mean():.1f}GB RAM",
                  delta=f"{df['storage_gb'].mean():.0f}GB ROM", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Charts ───
    render_section_header("🎨", "Biểu đồ tổng quan")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("##### 💰 Giá trung bình theo hãng")
        brand_avg = df.groupby("brand")["price_million_vnd"].mean().sort_values(ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=brand_avg.index, x=brand_avg.values, orientation="h",
            marker=dict(color=[BRAND_COLORS.get(b, COLORS["primary"]) for b in brand_avg.index]),
            text=[f"{v:.1f}tr" for v in brand_avg.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Giá TB: %{x:.2f} triệu VND<extra></extra>",
        ))
        fig.update_layout(height=400, margin=dict(l=0, r=20, t=10, b=0),
                          xaxis_title="Giá trung bình (triệu VND)", yaxis_title="",
                          plot_bgcolor="white", xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("##### 🥧 Thị phần theo số lượng")
        brand_count = df["brand"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=brand_count.index, values=brand_count.values, hole=0.55,
            marker=dict(colors=[BRAND_COLORS.get(b, COLORS["primary"]) for b in brand_count.index]),
            textinfo="label+percent", textfont=dict(size=11),
        )])
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            annotations=[dict(text=f"<b>{len(df)}</b><br>máy", x=0.5, y=0.5,
                              font_size=18, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("##### 📊 Phân bố giá")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["price_million_vnd"], nbinsx=30,
            marker=dict(color=COLORS["primary"], line=dict(color="white", width=1)),
        ))
        fig.add_vline(x=df["price_million_vnd"].mean(),
                      line=dict(color=COLORS["danger"], width=2, dash="dash"),
                      annotation_text=f"TB: {df['price_million_vnd'].mean():.1f}tr")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Giá (triệu VND)", yaxis_title="Số lượng",
                          plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("##### 🔗 Giá vs RAM")
        fig = px.scatter(df, x="ram_gb", y="price_million_vnd",
                         color="brand", size="storage_gb",
                         color_discrete_map=BRAND_COLORS,
                         hover_data={"model": True, "camera_mp": True})
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white"), opacity=0.7))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)

    # ─── Insights ───
    render_section_header("💡", "Insights nổi bật")
    c1, c2 = st.columns(2)
    with c1:
        top_expensive = df.groupby("brand")["price_million_vnd"].mean().idxmax()
        top_expensive_price = df.groupby("brand")["price_million_vnd"].mean().max()
        cheapest_brand = df.groupby("brand")["price_million_vnd"].mean().idxmin()
        cheapest_brand_price = df.groupby("brand")["price_million_vnd"].mean().min()

        render_info_box(f"🥇 <b>{top_expensive}</b> có giá TB cao nhất: "
                        f"{top_expensive_price:.2f} triệu VND", kind="info")
        render_info_box(f"💸 <b>{cheapest_brand}</b> có giá TB thấp nhất: "
                        f"{cheapest_brand_price:.2f} triệu VND", kind="success")

    with c2:
        numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores"]
        corrs = df[numeric_cols + ["price_million_vnd"]].corr()["price_million_vnd"].drop("price_million_vnd")
        strongest = corrs.abs().idxmax()
        render_info_box(f"⭐ <b>{strongest}</b> ảnh hưởng mạnh nhất đến giá "
                        f"(hệ số: {corrs[strongest]:+.3f})", kind="warning")
        premium_threshold = df["price_million_vnd"].quantile(0.9)
        premium_count = (df["price_million_vnd"] >= premium_threshold).sum()
        render_info_box(f"👑 <b>{premium_count} điện thoại</b> thuộc phân khúc cao cấp "
                        f"(≥ {premium_threshold:.2f} triệu VND)", kind="info")


# ═══════════════════════════════════════════════════════════════
#                   2) KHÁM PHÁ DỮ LIỆU
# ═══════════════════════════════════════════════════════════════
elif PAGE == "📊 Khám phá Dữ liệu":
    render_page_header(
        icon="📊",
        title="Khám phá Dữ liệu",
        subtitle="Lọc, tìm kiếm và khám phá dữ liệu điện thoại chi tiết",
    )

    # ─── Sidebar filters ───
    st.sidebar.markdown("### 🔍 Bộ lọc")
    selected_brands = st.sidebar.multiselect(
        "🏷️ Hãng", sorted(df["brand"].unique()), default=sorted(df["brand"].unique()),
    )
    pmin, pmax = float(df["price_million_vnd"].min()), float(df["price_million_vnd"].max())
    price_range = st.sidebar.slider("💰 Giá (triệu VND)", pmin, pmax, (pmin, pmax), step=0.5)

    rams = sorted(df["ram_gb"].unique())
    selected_rams = st.sidebar.multiselect("🧠 RAM (GB)", rams, default=rams)

    storages = sorted(df["storage_gb"].unique())
    selected_storage = st.sidebar.multiselect("💾 Bộ nhớ (GB)", storages, default=storages)

    cmin, cmax = int(df["camera_mp"].min()), int(df["camera_mp"].max())
    camera_range = st.sidebar.slider("📸 Camera (MP)", cmin, cmax, (cmin, cmax))

    bmin, bmax = int(df["battery_mah"].min()), int(df["battery_mah"].max())
    battery_range = st.sidebar.slider("🔋 Pin (mAh)", bmin, bmax, (bmin, bmax), step=100)

    # ─── Apply ───
    filtered = df[
        (df["brand"].isin(selected_brands)) &
        (df["price_million_vnd"].between(*price_range)) &
        (df["ram_gb"].isin(selected_rams)) &
        (df["storage_gb"].isin(selected_storage)) &
        (df["camera_mp"].between(*camera_range)) &
        (df["battery_mah"].between(*battery_range))
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Kết quả", f"{len(filtered):,}", delta=f"/{len(df):,} tổng")
    c2.metric("💰 Giá TB", f"{filtered['price_million_vnd'].mean():.2f}tr" if len(filtered) else "—")
    c3.metric("📉 Thấp nhất", f"{filtered['price_million_vnd'].min():.2f}tr" if len(filtered) else "—")
    c4.metric("📈 Cao nhất", f"{filtered['price_million_vnd'].max():.2f}tr" if len(filtered) else "—")

    if len(filtered) == 0:
        render_info_box("⚠️ Không có điện thoại nào thỏa mãn bộ lọc.", kind="warning")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Bảng", "📊 Thống kê", "📉 Phân bố", "🔍 Raw vs Clean"])

        with tab1:
            render_section_header("📋", "Bảng dữ liệu sau khi lọc")
            col1, col2 = st.columns([3, 1])
            with col1:
                search = st.text_input("🔎 Tìm model", placeholder="VD: Galaxy, iPhone...")
            with col2:
                sort_by = st.selectbox(
                    "⬇️ Sắp xếp",
                    ["price_million_vnd", "ram_gb", "storage_gb", "camera_mp", "battery_mah"],
                    format_func=lambda x: {
                        "price_million_vnd": "Giá", "ram_gb": "RAM", "storage_gb": "Bộ nhớ",
                        "camera_mp": "Camera", "battery_mah": "Pin",
                    }[x],
                )

            disp = filtered.copy()
            if search:
                disp = disp[disp["model"].str.contains(search, case=False, na=False)]
            disp = disp.sort_values(sort_by, ascending=False)

            disp_fmt = disp.rename(columns={
                "brand": "Hãng", "model": "Model",
                "ram_gb": "RAM (GB)", "storage_gb": "Bộ nhớ (GB)",
                "battery_mah": "Pin (mAh)", "camera_mp": "Camera (MP)",
                "screen_inch": "Màn hình (inch)", "cpu_cores": "Số nhân CPU",
                "price_million_vnd": "Giá (triệu VNĐ)",
            })
            st.dataframe(disp_fmt, use_container_width=True, height=500, hide_index=True)

            csv = disp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Tải CSV", csv, f"mobile_filtered_{len(disp)}.csv", "text/csv")

        with tab2:
            render_section_header("📊", "Thống kê mô tả")
            st.markdown("##### 📐 Các biến số")
            numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                            "screen_inch", "cpu_cores", "price_million_vnd"]
            stats = filtered[numeric_cols].describe().round(2)
            stats.index = ["Số lượng", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất",
                           "Q1 (25%)", "Trung vị (50%)", "Q3 (75%)", "Lớn nhất"]
            stats.columns = ["RAM", "Bộ nhớ", "Pin", "Camera", "Màn hình", "CPU cores", "Giá"]
            st.dataframe(stats, use_container_width=True)

            st.markdown("##### 🏷️ Theo hãng")
            bd = filtered["brand"].value_counts().reset_index()
            bd.columns = ["Hãng", "Số lượng"]
            bd["Tỷ lệ (%)"] = (bd["Số lượng"] / len(filtered) * 100).round(2)
            st.dataframe(bd, use_container_width=True, hide_index=True)

        with tab3:
            render_section_header("📉", "Phân bố các thuộc tính")
            col = st.selectbox(
                "Chọn thuộc tính:",
                ["price_million_vnd", "ram_gb", "storage_gb", "battery_mah",
                 "camera_mp", "screen_inch", "cpu_cores"],
                format_func=lambda x: {
                    "price_million_vnd": "💰 Giá", "ram_gb": "🧠 RAM",
                    "storage_gb": "💾 Bộ nhớ", "battery_mah": "🔋 Pin",
                    "camera_mp": "📸 Camera", "screen_inch": "📱 Màn hình",
                    "cpu_cores": "⚙️ CPU cores",
                }[x],
            )
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(filtered, x=col, nbins=30,
                                    color_discrete_sequence=[COLORS["primary"]], marginal="box")
                fig.update_layout(title="Histogram + Box", height=400, plot_bgcolor="white",
                                  xaxis=dict(gridcolor="#f1f5f9"),
                                  yaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(filtered, x="brand", y=col, color="brand",
                             color_discrete_map=BRAND_COLORS)
                fig.update_layout(title="Box plot theo hãng", height=400, showlegend=False,
                                  plot_bgcolor="white",
                                  xaxis=dict(gridcolor="#f1f5f9", tickangle=-30),
                                  yaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            render_section_header("🔍", "Dữ liệu thô vs Dữ liệu đã làm sạch")
            raw_df = load_raw_data()
            if raw_df.empty:
                render_info_box("⚠️ Chạy `python -m src.data_generator` trước.", kind="warning")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("📥 Dữ liệu thô", f"{len(raw_df):,}")
                c2.metric("✨ Dữ liệu sạch", f"{len(df):,}")
                diff = len(raw_df) - len(df)
                c3.metric("🧹 Đã xử lý", f"{diff:,}",
                          delta=f"-{diff/len(raw_df)*100:.1f}%", delta_color="inverse")

                render_info_box("ℹ️ Pipeline: chuẩn hóa đơn vị → điền missing → loại outlier → "
                                "loại trùng lặp.", kind="info")

                st.markdown("##### 🗂️ Giá trị thiếu trong raw data")
                missing = raw_df.isnull().sum()
                mdf = pd.DataFrame({
                    "Cột": missing.index, "Số thiếu": missing.values,
                    "Tỷ lệ (%)": (missing.values / len(raw_df) * 100).round(2),
                })
                mdf = mdf[mdf["Số thiếu"] > 0].sort_values("Số thiếu", ascending=False)
                if len(mdf) > 0:
                    st.dataframe(mdf, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Không có giá trị thiếu!")


# ═══════════════════════════════════════════════════════════════
#                3) PHÂN TÍCH & INSIGHT
# ═══════════════════════════════════════════════════════════════
elif PAGE == "📈 Phân tích & Insight":
    render_page_header(
        icon="📈",
        title="Phân tích & Insight",
        subtitle="Đi sâu vào dữ liệu để khám phá các yếu tố ảnh hưởng đến giá",
    )

    st.sidebar.markdown("### 🎯 Chọn phân tích")
    analysis_type = st.sidebar.radio(
        "Loại phân tích:",
        ["🏷️ Theo hãng", "⚙️ Theo cấu hình", "🔗 Ma trận tương quan",
         "🎚️ Phân khúc giá", "👑 Đặc điểm cao cấp"],
    )

    # ─── Theo hãng ───
    if analysis_type == "🏷️ Theo hãng":
        render_section_header("🏷️", "Phân tích theo hãng sản xuất")

        stats = df.groupby("brand").agg(
            so_luong=("price_million_vnd", "count"),
            gia_tb=("price_million_vnd", "mean"),
            gia_median=("price_million_vnd", "median"),
            gia_min=("price_million_vnd", "min"),
            gia_max=("price_million_vnd", "max"),
            ram_tb=("ram_gb", "mean"),
            storage_tb=("storage_gb", "mean"),
        ).round(2).sort_values("gia_tb", ascending=False)

        st.markdown("##### 📋 Bảng thống kê")
        disp = stats.reset_index().rename(columns={
            "brand": "Hãng", "so_luong": "Số lượng",
            "gia_tb": "Giá TB (tr)", "gia_median": "Median (tr)",
            "gia_min": "Min (tr)", "gia_max": "Max (tr)",
            "ram_tb": "RAM TB", "storage_tb": "Bộ nhớ TB",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 💰 Giá trung bình")
            fig = go.Figure(go.Bar(
                x=stats.index, y=stats["gia_tb"],
                marker=dict(color=[BRAND_COLORS.get(b, COLORS["primary"]) for b in stats.index]),
                text=[f"{v:.1f}" for v in stats["gia_tb"]], textposition="outside",
            ))
            fig.update_layout(height=400, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9", tickangle=-30),
                              yaxis=dict(gridcolor="#f1f5f9", title="Giá TB (triệu VND)"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### 📊 Box plot")
            fig = px.box(df.sort_values("brand"), x="brand", y="price_million_vnd",
                         color="brand", color_discrete_map=BRAND_COLORS)
            fig.update_layout(height=400, showlegend=False, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9", tickangle=-30, title=""),
                              yaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VND)"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        render_info_box(f"🥇 <b>{stats.index[0]}</b> có giá TB cao nhất: "
                        f"{stats.iloc[0]['gia_tb']:.2f} triệu", kind="info")
        render_info_box(f"💸 <b>{stats.index[-1]}</b> có giá TB thấp nhất: "
                        f"{stats.iloc[-1]['gia_tb']:.2f} triệu", kind="success")

    # ─── Theo cấu hình ───
    elif analysis_type == "⚙️ Theo cấu hình":
        render_section_header("⚙️", "Phân tích theo cấu hình")

        feature = st.selectbox(
            "Chọn cấu hình:",
            ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores"],
            format_func=lambda x: {
                "ram_gb": "🧠 RAM (GB)", "storage_gb": "💾 Bộ nhớ (GB)",
                "battery_mah": "🔋 Pin (mAh)", "camera_mp": "📸 Camera (MP)",
                "screen_inch": "📱 Màn hình (inch)", "cpu_cores": "⚙️ CPU cores",
            }[x],
        )
        label = {
            "ram_gb": "RAM (GB)", "storage_gb": "Bộ nhớ (GB)", "battery_mah": "Pin (mAh)",
            "camera_mp": "Camera (MP)", "screen_inch": "Màn hình (inch)", "cpu_cores": "CPU cores",
        }[feature]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"##### 📈 Giá TB theo {label}")
            fs = df.groupby(feature)["price_million_vnd"].mean().sort_index()
            fig = go.Figure(go.Bar(
                x=fs.index.astype(str), y=fs.values,
                marker=dict(color=fs.values, colorscale="Viridis", showscale=False),
                text=[f"{v:.1f}" for v in fs.values], textposition="outside",
            ))
            fig.update_layout(height=400, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9", title=label),
                              yaxis=dict(gridcolor="#f1f5f9", title="Giá TB (triệu VND)"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(f"##### 🔗 Scatter: {label} vs Giá")
            fig = px.scatter(df, x=feature, y="price_million_vnd",
                             color="brand", color_discrete_map=BRAND_COLORS,
                             hover_data={"model": True})
            add_linear_trendline(fig, df, feature)
            fig.update_traces(marker=dict(size=7, opacity=0.6, line=dict(width=0.5, color="white")))
            fig.update_layout(height=400, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9", title=label),
                              yaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VND)"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        corr = df[[feature, "price_million_vnd"]].corr().iloc[0, 1]
        strength = ("rất mạnh" if abs(corr) > 0.7 else "mạnh" if abs(corr) > 0.4
                    else "trung bình" if abs(corr) > 0.2 else "yếu")
        direction = "thuận" if corr > 0 else "nghịch"
        render_info_box(f"📊 Pearson giữa <b>{label}</b> và giá: <b>{corr:+.3f}</b> "
                        f"(tương quan {direction}, độ mạnh {strength})", kind="info")

    # ─── Ma trận tương quan ───
    elif analysis_type == "🔗 Ma trận tương quan":
        render_section_header("🔗", "Ma trận tương quan các thuộc tính")

        numeric_cols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                        "screen_inch", "cpu_cores", "price_million_vnd"]
        label_map = {"ram_gb": "RAM", "storage_gb": "Bộ nhớ", "battery_mah": "Pin",
                     "camera_mp": "Camera", "screen_inch": "Màn hình",
                     "cpu_cores": "CPU cores", "price_million_vnd": "Giá"}
        corr = df[numeric_cols].corr()
        corr.index = [label_map[c] for c in corr.index]
        corr.columns = [label_map[c] for c in corr.columns]

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}", textfont={"size": 13},
        ))
        fig.update_layout(height=520, plot_bgcolor="white", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### 🎯 Mức độ ảnh hưởng đến giá")
        pc = corr["Giá"].drop("Giá").sort_values(ascending=False)
        cdf = pd.DataFrame({
            "Thuộc tính": pc.index, "Hệ số": pc.values,
            "Ảnh hưởng": ["⬆️ Mạnh" if abs(v) > 0.5 else "↗️ TB" if abs(v) > 0.2 else "➡️ Yếu"
                          for v in pc.values],
        })
        st.dataframe(cdf, use_container_width=True, hide_index=True,
                     column_config={"Hệ số": st.column_config.ProgressColumn(
                         "Hệ số", min_value=-1, max_value=1, format="%.3f")})

        strongest = pc.abs().idxmax()
        render_info_box(f"⭐ <b>{strongest}</b> ảnh hưởng mạnh nhất "
                        f"(hệ số {pc[strongest]:+.3f})", kind="warning")

    # ─── Phân khúc giá ───
    elif analysis_type == "🎚️ Phân khúc giá":
        render_section_header("🎚️", "Phân khúc giá")

        df_seg = df.copy()
        df_seg["segment"] = pd.qcut(df_seg["price_million_vnd"], q=[0, 0.33, 0.66, 1.0],
                                     labels=["Giá rẻ", "Tầm trung", "Cao cấp"])

        seg_stats = df_seg.groupby("segment", observed=True).agg(
            so_luong=("price_million_vnd", "count"),
            gia_min=("price_million_vnd", "min"),
            gia_max=("price_million_vnd", "max"),
            gia_tb=("price_million_vnd", "mean"),
            ram=("ram_gb", "mean"),
            storage=("storage_gb", "mean"),
            camera=("camera_mp", "mean"),
        ).round(2)

        c1, c2, c3 = st.columns(3)
        seg_colors = ["#10b981", "#f59e0b", "#ef4444"]
        seg_icons = ["💵", "💰", "💎"]
        for (seg, color, icon), col in zip(
            zip(["Giá rẻ", "Tầm trung", "Cao cấp"], seg_colors, seg_icons),
            [c1, c2, c3]
        ):
            if seg in seg_stats.index:
                r = seg_stats.loc[seg]
                with col:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color}22, {color}11);
                                padding: 1.5rem; border-radius: 14px;
                                border-left: 5px solid {color};'>
                        <div style='font-size: 2rem;'>{icon}</div>
                        <h3 style='color: {color}; margin: 0.5rem 0;'>{seg}</h3>
                        <p style='font-size: 1.5rem; font-weight: 700; margin: 0;'>
                            {r['gia_min']:.1f} - {r['gia_max']:.1f} tr
                        </p>
                        <p style='color: #64748b; margin: 0.5rem 0 0 0;'>
                            {int(r['so_luong']):,} máy • TB {r['gia_tb']:.1f} tr
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ⚙️ Cấu hình TB theo phân khúc")
            cdf = seg_stats[["ram", "storage", "camera"]].T
            cdf.index = ["RAM (GB)", "Bộ nhớ (GB)", "Camera (MP)"]
            fig = go.Figure()
            for seg, color in zip(cdf.columns, seg_colors):
                fig.add_trace(go.Bar(
                    name=seg, x=cdf.index, y=cdf[seg], marker=dict(color=color),
                    text=[f"{v:.1f}" for v in cdf[seg]], textposition="outside",
                ))
            fig.update_layout(height=400, barmode="group", plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9"), yaxis=dict(gridcolor="#f1f5f9"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### 🏭 Phân bố hãng theo phân khúc")
            bs = df_seg.groupby(["segment", "brand"], observed=True).size().reset_index(name="count")
            fig = px.sunburst(bs, path=["segment", "brand"], values="count",
                              color="segment",
                              color_discrete_map={"Giá rẻ": seg_colors[0],
                                                  "Tầm trung": seg_colors[1],
                                                  "Cao cấp": seg_colors[2]})
            fig.update_layout(height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ─── Đặc điểm cao cấp ───
    elif analysis_type == "👑 Đặc điểm cao cấp":
        render_section_header("👑", "Đặc điểm điện thoại cao cấp (Top 10%)")

        threshold = df["price_million_vnd"].quantile(0.9)
        premium = df[df["price_million_vnd"] >= threshold]
        regular = df[df["price_million_vnd"] < threshold]

        c1, c2, c3 = st.columns(3)
        c1.metric("💎 Ngưỡng premium", f"{threshold:.2f} tr")
        c2.metric("🏆 Số điện thoại", f"{len(premium):,}")
        c3.metric("📊 Tỷ lệ", f"{len(premium)/len(df)*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📏 So sánh Premium vs Regular")

        features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch"]
        labels = ["RAM (GB)", "Bộ nhớ (GB)", "Pin (mAh)", "Camera (MP)", "Màn hình (inch)"]
        comp = pd.DataFrame([{
            "Thuộc tính": label,
            "Premium (TB)": premium[f].mean(),
            "Regular (TB)": regular[f].mean(),
            "Chênh (%)": (premium[f].mean() - regular[f].mean()) / regular[f].mean() * 100,
        } for f, label in zip(features, labels)])

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Premium", x=comp["Thuộc tính"], y=comp["Premium (TB)"],
                                 marker=dict(color="#6366f1")))
            fig.add_trace(go.Bar(name="Regular", x=comp["Thuộc tính"], y=comp["Regular (TB)"],
                                 marker=dict(color="#cbd5e1")))
            fig.update_layout(height=400, barmode="group", plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9"),
                              yaxis=dict(gridcolor="#f1f5f9", title="Giá trị trung bình"),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("##### 🔝 Top hãng premium")
            tb = premium["brand"].value_counts().head(5).reset_index()
            tb.columns = ["Hãng", "Số lượng"]
            tb["Tỷ lệ"] = (tb["Số lượng"] / len(premium) * 100).round(1).astype(str) + "%"
            st.dataframe(tb, use_container_width=True, hide_index=True)

        comp_disp = comp.copy()
        comp_disp["Premium (TB)"] = comp_disp["Premium (TB)"].round(2)
        comp_disp["Regular (TB)"] = comp_disp["Regular (TB)"].round(2)
        comp_disp["Chênh (%)"] = comp_disp["Chênh (%)"].round(1).astype(str) + "%"
        st.dataframe(comp_disp, use_container_width=True, hide_index=True)

        render_info_box("💡 <b>Kết luận:</b> Điện thoại cao cấp có RAM, bộ nhớ và camera "
                        "vượt trội. Apple/Samsung/Google thống trị phân khúc này.", kind="info")


# ═══════════════════════════════════════════════════════════════
#                  4) BIỂU ĐỒ TRỰC QUAN
# ═══════════════════════════════════════════════════════════════
elif PAGE == "🎨 Biểu đồ Trực quan":
    render_page_header(
        icon="🎨",
        title="Biểu đồ Trực quan",
        subtitle="Khám phá dữ liệu qua các biểu đồ tương tác có thể tùy chỉnh",
    )

    st.sidebar.markdown("### 🎨 Chọn biểu đồ")
    chart_type = st.sidebar.radio(
        "Loại biểu đồ:",
        ["📊 Biểu đồ cột - Giá theo hãng", "🔗 Biểu đồ phân tán - Giá và cấu hình",
         "🌡️ Bản đồ nhiệt - Tương quan", "📈 Biểu đồ tần suất - Phân bố giá",
         "📦 Biểu đồ hộp - Theo hãng", "🫧 Biểu đồ bong bóng - Giá/RAM/Bộ nhớ",
         "🌐 Biểu đồ phân tán 3D", "🎯 Biểu đồ radar - So sánh hãng",
         "🔥 Biểu đồ vòng phân cấp", "📊 Hãng x Phân khúc giá",
         "🖼️ Ảnh biểu đồ báo cáo", "⭐ Tầm quan trọng đặc trưng",
         "🎯 Giá thực tế và giá dự đoán"],
    )

    # Bar chart
    if chart_type == "📊 Biểu đồ cột - Giá theo hãng":
        render_section_header("📊", "Biểu đồ cột: Giá theo hãng")
        c1, c2 = st.columns([1, 3])
        with c1:
            agg = st.selectbox("Kiểu:", ["Trung bình", "Trung vị", "Min", "Max"])
            orient = st.radio("Hướng:", ["Ngang", "Dọc"])
            order = st.radio("Sắp xếp:", ["Giảm dần", "Tăng dần", "Theo tên"])
        with c2:
            amap = {"Trung bình": "mean", "Trung vị": "median", "Min": "min", "Max": "max"}
            bd = df.groupby("brand")["price_million_vnd"].agg(amap[agg])
            if order == "Giảm dần": bd = bd.sort_values(ascending=False)
            elif order == "Tăng dần": bd = bd.sort_values(ascending=True)
            else: bd = bd.sort_index()

            colors = [BRAND_COLORS.get(b, COLORS["primary"]) for b in bd.index]
            if orient == "Ngang":
                fig = go.Figure(go.Bar(y=bd.index, x=bd.values, orientation="h",
                                       marker=dict(color=colors),
                                       text=[f"{v:.2f}" for v in bd.values],
                                       textposition="outside"))
                fig.update_layout(xaxis_title=f"Giá {agg.lower()} (triệu VND)")
            else:
                fig = go.Figure(go.Bar(x=bd.index, y=bd.values, marker=dict(color=colors),
                                       text=[f"{v:.2f}" for v in bd.values],
                                       textposition="outside"))
                fig.update_layout(yaxis_title=f"Giá {agg.lower()} (triệu VND)")
            fig.update_layout(height=500, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9"),
                              yaxis=dict(gridcolor="#f1f5f9"))
            st.plotly_chart(fig, use_container_width=True)
            top_brand, top_value = bd.idxmax(), bd.max()
            low_brand, low_value = bd.idxmin(), bd.min()
            render_chart_comment(
                f"{top_brand} có giá {agg.lower()} cao nhất ({top_value:.2f} triệu VND), "
                f"trong khi {low_brand} thấp nhất ({low_value:.2f} triệu VND). "
                "Biểu đồ giúp so sánh định vị giá giữa các hãng."
            )

    # Scatter
    elif chart_type == "🔗 Biểu đồ phân tán - Giá và cấu hình":
        render_section_header("🔗", "Biểu đồ phân tán: Giá và cấu hình")
        c1, c2, c3 = st.columns(3)
        with c1:
            x_feat = st.selectbox("Trục X:",
                ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch"],
                format_func=lambda x: {"ram_gb": "RAM", "storage_gb": "Bộ nhớ",
                                        "battery_mah": "Pin", "camera_mp": "Camera",
                                        "screen_inch": "Màn hình"}[x])
        with c2:
            color_by = st.selectbox("Màu theo:", ["brand", "cpu_cores"],
                                     format_func=lambda x: {"brand": "Hãng", "cpu_cores": "CPU"}[x])
        with c3:
            trend = st.checkbox("Đường xu hướng", value=True)

        cmap = BRAND_COLORS if color_by == "brand" else None
        fig = px.scatter(df, x=x_feat, y="price_million_vnd", color=color_by,
                         color_discrete_map=cmap, hover_data={"model": True, "brand": True})
        if trend:
            add_linear_trendline(fig, df, x_feat)
        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="white")))
        fig.update_layout(height=550, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title=vn_feature_label(x_feat)),
                          yaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VND)"))
        st.plotly_chart(fig, use_container_width=True)
        corr_value = df[[x_feat, "price_million_vnd"]].corr().iloc[0, 1]
        direction = "cùng chiều" if corr_value >= 0 else "ngược chiều"
        render_chart_comment(
            f"{vn_feature_label(x_feat)} có tương quan {direction} {corr_strength(corr_value)} "
            f"với giá (hệ số {corr_value:.2f}). Các điểm nằm cao hơn thể hiện những mẫu có giá cao hơn "
            "ở cùng mức cấu hình."
        )

    # Heatmap
    elif chart_type == "🌡️ Bản đồ nhiệt - Tương quan":
        render_section_header("🌡️", "Bản đồ nhiệt: Ma trận tương quan")
        ncols = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                 "screen_inch", "cpu_cores", "price_million_vnd"]
        lm = {"ram_gb": "RAM", "storage_gb": "Bộ nhớ", "battery_mah": "Pin",
              "camera_mp": "Camera", "screen_inch": "Màn hình",
              "cpu_cores": "CPU", "price_million_vnd": "Giá"}
        method = st.radio("Phương pháp:", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = df[ncols].corr(method=method)
        corr.index = [lm[c] for c in corr.index]
        corr.columns = [lm[c] for c in corr.columns]

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=corr.round(3).values, texttemplate="%{text}", textfont={"size": 14},
        ))
        fig.update_layout(height=550, plot_bgcolor="white", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        price_corr = corr["Giá"].drop("Giá").abs().sort_values(ascending=False)
        render_chart_comment(
            f"{price_corr.index[0]} là thuộc tính có liên hệ mạnh nhất với giá trong ma trận này "
            f"(hệ số {price_corr.iloc[0]:.2f}). Màu càng đậm thì mức liên hệ giữa hai thuộc tính càng rõ."
        )

        heatmap_image = Path("correlation_heatmap.png")
        if not heatmap_image.exists():
            heatmap_image = Path("outputs/figures/03_correlation_heatmap.png")

        if heatmap_image.exists():
            render_section_header("🖼️", "Ảnh heatmap để chèn báo cáo")
            st.image(str(heatmap_image), caption="Ma trận tương quan", use_container_width=True)
            with open(heatmap_image, "rb") as f:
                st.download_button(
                    "⬇️ Tải ảnh heatmap PNG",
                    data=f,
                    file_name="correlation_heatmap.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            render_info_box("⚠️ Chưa tìm thấy ảnh heatmap PNG. Chạy `python main.py` để tạo ảnh.", kind="warning")

    # Histogram
    elif chart_type == "📈 Biểu đồ tần suất - Phân bố giá":
        render_section_header("📈", "Biểu đồ tần suất: Phân bố giá")
        c1, c2 = st.columns([1, 3])
        with c1:
            bins = st.slider("Số bin:", 10, 100, 40, step=5)
            kde = st.checkbox("KDE", value=True)
            by_brand = st.checkbox("Phân theo hãng", value=False)
        with c2:
            if by_brand:
                fig = px.histogram(df, x="price_million_vnd", color="brand",
                                    nbins=bins, color_discrete_map=BRAND_COLORS,
                                    marginal="box" if kde else None,
                                    barmode="overlay", opacity=0.6)
            else:
                fig = px.histogram(df, x="price_million_vnd", nbins=bins,
                                    color_discrete_sequence=[COLORS["primary"]],
                                    marginal="violin" if kde else None)
                fig.add_vline(x=df["price_million_vnd"].mean(),
                              line=dict(color=COLORS["danger"], dash="dash", width=2),
                              annotation_text=f"TB: {df['price_million_vnd'].mean():.2f}")
            fig.update_layout(height=550, plot_bgcolor="white",
                              xaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VND)"),
                              yaxis=dict(gridcolor="#f1f5f9", title="Số lượng"))
            st.plotly_chart(fig, use_container_width=True)
            mean_price = df["price_million_vnd"].mean()
            median_price = df["price_million_vnd"].median()
            render_chart_comment(
                f"Giá trung bình là {mean_price:.2f} triệu VND và trung vị là {median_price:.2f} triệu VND. "
                "Nếu trung bình cao hơn trung vị, dữ liệu có xu hướng bị kéo lên bởi các mẫu giá cao."
            )

    # Box plot
    elif chart_type == "📦 Biểu đồ hộp - Theo hãng":
        render_section_header("📦", "Biểu đồ hộp: So sánh theo hãng")
        feat = st.selectbox("Thuộc tính:",
            ["price_million_vnd", "ram_gb", "storage_gb", "battery_mah", "camera_mp"],
            format_func=lambda x: {"price_million_vnd": "Giá", "ram_gb": "RAM",
                                    "storage_gb": "Bộ nhớ", "battery_mah": "Pin",
                                    "camera_mp": "Camera"}[x])
        fig = px.box(df.sort_values("brand"), x="brand", y=feat, color="brand",
                     color_discrete_map=BRAND_COLORS, points="outliers")
        fig.update_layout(height=550, showlegend=False, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", tickangle=-30, title=""),
                          yaxis=dict(gridcolor="#f1f5f9", title=vn_feature_label(feat)))
        st.plotly_chart(fig, use_container_width=True)
        medians = df.groupby("brand")[feat].median().sort_values(ascending=False)
        q = df.groupby("brand")[feat].quantile([0.25, 0.75]).unstack()
        widest_brand = (q[0.75] - q[0.25]).idxmax()
        render_chart_comment(
            f"{medians.index[0]} có trung vị {vn_feature_label(feat).lower()} cao nhất "
            f"({medians.iloc[0]:.2f}). {widest_brand} có khoảng biến thiên giữa các mẫu lớn nhất, "
            "cho thấy danh mục sản phẩm phân tán hơn."
        )

    # Bubble chart
    elif chart_type == "🫧 Biểu đồ bong bóng - Giá/RAM/Bộ nhớ":
        render_section_header("🫧", "Biểu đồ bong bóng: Giá, RAM và bộ nhớ")
        fig = px.scatter(df, x="ram_gb", y="price_million_vnd", size="storage_gb",
                         color="brand", color_discrete_map=BRAND_COLORS,
                         hover_data={"model": True, "camera_mp": True},
                         size_max=40)
        fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.75))
        fig.update_layout(height=600, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="RAM (GB)"),
                          yaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VND)"))
        st.plotly_chart(fig, use_container_width=True)
        storage_corr = df[["storage_gb", "price_million_vnd"]].corr().iloc[0, 1]
        render_chart_comment(
            f"Kích thước bong bóng thể hiện dung lượng bộ nhớ. Bộ nhớ có tương quan "
            f"{corr_strength(storage_corr)} với giá (hệ số {storage_corr:.2f}); các bong bóng lớn ở vùng giá cao "
            "thường là những mẫu có cấu hình lưu trữ tốt hơn."
        )

    # 3D Scatter
    elif chart_type == "🌐 Biểu đồ phân tán 3D":
        render_section_header("🌐", "Biểu đồ phân tán 3D")
        c1, c2, c3 = st.columns(3)
        with c1:
            x = st.selectbox("X:", ["ram_gb", "storage_gb", "battery_mah", "camera_mp"])
        with c2:
            y = st.selectbox("Y:", ["storage_gb", "ram_gb", "battery_mah", "camera_mp"])
        with c3:
            z = st.selectbox("Z:", ["price_million_vnd"])
        fig = px.scatter_3d(df, x=x, y=y, z=z, color="brand",
                            color_discrete_map=BRAND_COLORS, hover_data={"model": True})
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
        x_corr = df[[x, "price_million_vnd"]].corr().iloc[0, 1]
        y_corr = df[[y, "price_million_vnd"]].corr().iloc[0, 1]
        stronger_axis = vn_feature_label(x if abs(x_corr) >= abs(y_corr) else y)
        render_chart_comment(
            f"Biểu đồ 3D giúp quan sát đồng thời {vn_feature_label(x)}, {vn_feature_label(y)} và giá. "
            f"Trong hai trục cấu hình đang chọn, {stronger_axis} liên hệ rõ hơn với giá."
        )

    # Radar
    elif chart_type == "🎯 Biểu đồ radar - So sánh hãng":
        render_section_header("🎯", "Biểu đồ radar: So sánh hãng")
        sel = st.multiselect("Chọn 2-5 hãng:", sorted(df["brand"].unique()),
                              default=["Apple", "Samsung", "Xiaomi"], max_selections=5)
        if len(sel) >= 2:
            feats = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "price_million_vnd"]
            labels = ["RAM", "Bộ nhớ", "Pin", "Camera", "Giá"]
            norm = df[feats].copy()
            for f in feats:
                norm[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())
            norm["brand"] = df["brand"]

            fig = go.Figure()
            for b in sel:
                vals = norm[norm["brand"] == b][feats].mean().tolist()
                vals += [vals[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=labels + [labels[0]],
                    fill="toself", name=b,
                    line=dict(color=BRAND_COLORS.get(b, COLORS["primary"]), width=2),
                    opacity=0.7,
                ))
            fig.update_layout(height=600, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
            selected_means = norm[norm["brand"].isin(sel)].groupby("brand")[feats].mean()
            best_price_brand = selected_means["price_million_vnd"].idxmax()
            render_chart_comment(
                f"Các giá trị đã được chuẩn hóa từ 0 đến 1; càng gần biên ngoài càng cao. "
                f"Trong nhóm đang chọn, {best_price_brand} có mặt bằng giá chuẩn hóa cao nhất."
            )
        else:
            st.warning("⚠️ Chọn ít nhất 2 hãng.")

    # Sunburst
    elif chart_type == "🔥 Biểu đồ vòng phân cấp":
        render_section_header("🔥", "Biểu đồ vòng phân cấp: Cấu trúc thị trường")
        ds = df.copy()
        ds["segment"] = pd.qcut(ds["price_million_vnd"], q=[0, 0.33, 0.66, 1.0],
                                 labels=["Giá rẻ", "Tầm trung", "Cao cấp"])
        ds["ram_label"] = ds["ram_gb"].astype(str) + " GB RAM"
        agg = ds.groupby(["segment", "brand", "ram_label"], observed=True).size().reset_index(name="count")
        fig = px.sunburst(agg, path=["segment", "brand", "ram_label"], values="count",
                          color="segment",
                          color_discrete_map={"Giá rẻ": "#10b981",
                                              "Tầm trung": "#f59e0b",
                                              "Cao cấp": "#ef4444"})
        fig.update_layout(height=650, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        segment_counts = ds["segment"].value_counts()
        brand_counts = ds.groupby(["segment", "brand"], observed=True).size().sort_values(ascending=False)
        top_segment, top_brand = brand_counts.index[0]
        render_chart_comment(
            f"Phân khúc {segment_counts.idxmax()} có nhiều mẫu nhất. Nhánh lớn nhất là {top_brand} "
            f"trong phân khúc {top_segment}, thể hiện nhóm sản phẩm đóng góp nhiều nhất vào cấu trúc hiện tại."
        )

    # Brand x Price Range
    elif chart_type == "📊 Hãng x Phân khúc giá":
        render_section_header("📊", "Số lượng điện thoại theo hãng và phân khúc giá")
        ds = df.copy()
        q33, q66 = ds["price_million_vnd"].quantile([0.33, 0.66])
        ds["price_range"] = pd.cut(
            ds["price_million_vnd"],
            bins=[-np.inf, q33, q66, np.inf],
            labels=["Giá rẻ", "Tầm trung", "Cao cấp"],
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            chart_mode = st.radio("Kiểu hiển thị:", ["Chồng", "Nhóm"], horizontal=False)
            sort_by = st.radio("Sắp xếp:", ["Tổng số", "Tên hãng"], horizontal=False)
        with c2:
            count_df = (
                ds.groupby(["brand", "price_range"], observed=True)
                .size()
                .reset_index(name="count")
            )
            brand_order = ds["brand"].value_counts().index.tolist()
            if sort_by == "Tên hãng":
                brand_order = sorted(brand_order)

            fig = px.bar(
                count_df,
                x="brand",
                y="count",
                color="price_range",
                category_orders={
                    "brand": brand_order,
                    "price_range": ["Giá rẻ", "Tầm trung", "Cao cấp"],
                },
                color_discrete_map={
                    "Giá rẻ": "#10b981",
                    "Tầm trung": "#f59e0b",
                    "Cao cấp": "#ef4444",
                },
                text="count",
                barmode="stack" if chart_mode == "Chồng" else "group",
            )
            fig.update_traces(textposition="outside" if chart_mode == "Nhóm" else "inside")
            fig.update_layout(
                height=560,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9", title="Hãng"),
                yaxis=dict(gridcolor="#f1f5f9", title="Số lượng điện thoại"),
                legend_title="Phân khúc giá",
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        render_info_box(
            f"📌 Phân khúc giá được chia theo dữ liệu hiện tại: Giá rẻ ≤ {q33:.2f} triệu, "
            f"Tầm trung từ {q33:.2f} đến {q66:.2f} triệu, Cao cấp > {q66:.2f} triệu.",
            kind="info",
        )
        top_count = count_df.sort_values("count", ascending=False).iloc[0]
        render_chart_comment(
            f"{top_count['brand']} có số lượng mẫu nhiều nhất ở phân khúc {top_count['price_range']} "
            f"({int(top_count['count'])} mẫu). Biểu đồ cho thấy mỗi hãng tập trung vào phân khúc giá nào."
        )

    # Report chart images
    elif chart_type == "🖼️ Ảnh biểu đồ báo cáo":
        render_section_header("🖼️", "Ảnh biểu đồ dùng cho báo cáo")
        image_items = [
            ("Giá trung bình theo hãng", Path("outputs/figures/01_avg_price_by_brand.png")),
            ("Giá vs cấu hình", Path("outputs/figures/02_scatter_price_vs_specs.png")),
            ("Ma trận tương quan", Path("correlation_heatmap.png")),
            ("Phân bố giá", Path("outputs/figures/04_price_distribution.png")),
            ("Phân tích phân khúc giá", Path("outputs/figures/05_segment_analysis.png")),
            ("So sánh mô hình", Path("outputs/figures/06_model_comparison.png")),
            ("Giá thực tế và giá dự đoán", Path("outputs/figures/07_predictions_vs_actual.png")),
            ("Tầm quan trọng đặc trưng", Path("outputs/figures/08_feature_importance.png")),
            ("Hãng x Phân khúc giá", Path("brand_price_range_barplot.png")),
        ]
        image_items = [(title, path) for title, path in image_items if path.exists()]

        if not image_items:
            render_info_box("⚠️ Chưa có ảnh biểu đồ. Hãy chạy `python main.py` để tạo ảnh.", kind="warning")
        else:
            selected_title = st.selectbox("Chọn ảnh biểu đồ:", [title for title, _ in image_items])
            selected_path = next(path for title, path in image_items if title == selected_title)
            st.image(str(selected_path), caption=selected_title, use_container_width=True)
            render_chart_comment(
                f"Ảnh '{selected_title}' là phiên bản tĩnh phù hợp để chèn vào báo cáo hoặc slide thuyết trình."
            )
            with open(selected_path, "rb") as f:
                st.download_button(
                    "⬇️ Tải ảnh PNG",
                    data=f,
                    file_name=selected_path.name,
                    mime="image/png",
                    use_container_width=True,
                )

    # Feature Importance
    elif chart_type == "⭐ Tầm quan trọng đặc trưng":
        render_section_header("⭐", "Tầm quan trọng đặc trưng: Yếu tố ảnh hưởng đến giá")
        model = load_model()
        if model is None:
            render_info_box("⚠️ Chưa có mô hình. Hãy chạy `python main.py` để huấn luyện trước.", kind="warning")
        else:
            importance_df = build_feature_importance(model, df)
            top_feature = importance_df.iloc[-1]

            fig = go.Figure(go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation="h",
                marker=dict(color=importance_df["importance"], colorscale="Viridis", showscale=False),
                text=[f"{v:.3f}" for v in importance_df["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=520,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9", title="Mức độ ảnh hưởng"),
                yaxis=dict(title=""),
                margin=dict(t=20, r=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            render_info_box(
                f"⭐ <b>{top_feature['feature']}</b> là yếu tố ảnh hưởng mạnh nhất đến giá "
                f"theo {top_feature['method'].lower()}.",
                kind="info",
            )
            render_chart_comment(
                "Thanh càng dài thì đặc trưng đó càng đóng góp nhiều hơn vào quyết định dự đoán giá của mô hình."
            )

    # Actual vs Predicted
    elif chart_type == "🎯 Giá thực tế và giá dự đoán":
        render_section_header("🎯", "Giá thực tế và giá dự đoán")
        model = load_model()
        if model is None:
            render_info_box("⚠️ Chưa có mô hình. Hãy chạy `python main.py` để huấn luyện trước.", kind="warning")
        else:
            X_test, y_test, y_pred, residuals = get_prediction_sample(model, df)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f} tr")
            c2.metric("RMSE", f"{rmse:.3f} tr")
            c3.metric("Số mẫu test", f"{len(y_test):,}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(
                    color=residuals,
                    colorscale="RdYlGn_r",
                    cmid=0,
                    size=8,
                    opacity=0.72,
                    line=dict(width=0.5, color="white"),
                    colorbar=dict(title="Sai số"),
                ),
                text=X_test["brand"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Giá thực tế: %{x:.2f} tr<br>"
                    "Giá dự đoán: %{y:.2f} tr<br>"
                    "Sai số: %{marker.color:.2f} tr<extra></extra>"
                ),
                name="Điện thoại",
            ))
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            fig.add_trace(go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(color=COLORS["danger"], dash="dash", width=2),
                name="Dự đoán hoàn hảo",
            ))
            fig.update_layout(
                height=560,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9", title="Giá thực tế (triệu VND)"),
                yaxis=dict(gridcolor="#f1f5f9", title="Giá dự đoán (triệu VND)"),
                margin=dict(t=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            render_info_box(
                "💡 Các điểm càng gần đường chéo màu đỏ thì mô hình dự đoán càng chính xác.",
                kind="info",
            )
            render_chart_comment(
                f"Sai số tuyệt đối trung bình hiện là {mae:.3f} triệu VND. "
                "Những điểm lệch xa đường chéo là các mẫu mô hình dự đoán chưa sát với giá thực tế."
            )


# ═══════════════════════════════════════════════════════════════
#                      5) DỰ ĐOÁN GIÁ
# ═══════════════════════════════════════════════════════════════
elif PAGE == "🔮 Dự đoán Giá":
    render_page_header(
        icon="🔮",
        title="Dự đoán Giá Điện thoại",
        subtitle="Nhập cấu hình điện thoại, AI sẽ dự đoán giá thị trường",
    )

    model = load_model()
    if model is None:
        render_info_box(
            "⚠️ Chưa có mô hình. Chạy `python main.py` trước hoặc nhấn nút bên dưới để huấn luyện ngay.",
            kind="warning",
        )
        if st.button("🔧 Huấn luyện mô hình ngay", use_container_width=True):
            with st.spinner("⏳ Đang huấn luyện mô hình..."):
                from src.model_training import run_training

                try:
                    run_training()
                    st.success(
                        "✅ Đã huấn luyện xong! Mô hình đã được lưu vào `outputs/models/best_model.pkl`."
                    )
                    st.cache_resource.clear()
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"❌ Huấn luyện thất bại: {e}")
            st.experimental_rerun()
        st.stop()

    st.sidebar.markdown("### 🎯 Chế độ")
    mode = st.sidebar.radio("Cách nhập:", ["🎛️ Tùy chỉnh", "📱 Preset"])
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Dữ liệu:** {len(df):,} điện thoại\n\n**Độ chính xác:** R² ~ 0.89")

    render_section_header("⚙️", "Thông số kỹ thuật")

    c1, c2 = st.columns(2)

    if mode == "📱 Preset":
        presets = {
            "Flagship Apple": {"brand": "Apple", "ram_gb": 8, "storage_gb": 256,
                               "battery_mah": 4500, "camera_mp": 48, "screen_inch": 6.1, "cpu_cores": 6},
            "Samsung Tầm trung": {"brand": "Samsung", "ram_gb": 6, "storage_gb": 128,
                                   "battery_mah": 5000, "camera_mp": 50, "screen_inch": 6.5, "cpu_cores": 8},
            "Xiaomi Giá rẻ": {"brand": "Xiaomi", "ram_gb": 4, "storage_gb": 64,
                              "battery_mah": 5000, "camera_mp": 48, "screen_inch": 6.5, "cpu_cores": 8},
            "Oppo Cao cấp": {"brand": "Oppo", "ram_gb": 12, "storage_gb": 256,
                             "battery_mah": 4800, "camera_mp": 108, "screen_inch": 6.7, "cpu_cores": 8},
            "Asus ROG": {"brand": "Asus", "ram_gb": 16, "storage_gb": 512,
                         "battery_mah": 6000, "camera_mp": 64, "screen_inch": 6.8, "cpu_cores": 8},
        }
        with c1:
            selected = st.selectbox("🎁 Chọn preset:", list(presets.keys()))
        specs = presets[selected].copy()
        input_source = selected
        with c2:
            st.markdown(f"##### 📝 Thông số {selected}")
            for k, v in specs.items():
                dk = {"brand": "Hãng", "ram_gb": "RAM", "storage_gb": "Bộ nhớ",
                      "battery_mah": "Pin", "camera_mp": "Camera",
                      "screen_inch": "Màn hình", "cpu_cores": "CPU cores"}[k]
                st.markdown(f"- **{dk}:** {v}")
    else:
        with c1:
            brand = st.selectbox("🏷️ Hãng", sorted(df["brand"].unique()))
            ram_gb = st.select_slider("🧠 RAM (GB)", [2, 3, 4, 6, 8, 12, 16], value=8)
            storage_gb = st.select_slider("💾 Bộ nhớ (GB)", [32, 64, 128, 256, 512, 1024], value=128)
            battery_mah = st.slider("🔋 Pin (mAh)", 3000, 6500, 4500, step=100)
        with c2:
            camera_mp = st.select_slider("📸 Camera (MP)",
                                          [8, 12, 13, 48, 50, 64, 108, 200], value=50)
            screen_inch = st.slider("📱 Màn hình (inch)", 5.0, 7.0, 6.5, step=0.1)
            cpu_cores = st.select_slider("⚙️ CPU cores", [4, 6, 8], value=8)

        specs = {"brand": brand, "ram_gb": ram_gb, "storage_gb": storage_gb,
                 "battery_mah": battery_mah, "camera_mp": camera_mp,
                 "screen_inch": screen_inch, "cpu_cores": cpu_cores}
        input_source = "Tùy chỉnh"

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 Dự đoán giá", use_container_width=True, type="primary")

    if predict_clicked or mode == "📱 Preset":
        X_pred = pd.DataFrame([specs])
        predicted_price = model.predict(X_pred)[0]

        render_section_header("💰", "Kết quả dự đoán")
        st.markdown(f"""
        <div class="predict-result fade-in">
            <div class="label">Giá dự đoán</div>
            <div>
                <span class="value">{predicted_price:.2f}</span>
                <span class="unit">triệu VNĐ</span>
            </div>
            <div style="margin-top: 0.75rem; font-size: 1rem; opacity: 0.9;">
                ≈ {predicted_price * 1_000_000:,.0f} VNĐ
            </div>
        </div>
        """, unsafe_allow_html=True)

        low, high = predicted_price * 0.85, predicted_price * 1.15
        c1, c2, c3 = st.columns(3)
        c1.metric("🔻 Ước lượng thấp", f"{low:.2f} tr", delta="-15%", delta_color="off")
        c2.metric("🎯 Dự đoán", f"{predicted_price:.2f} tr", delta="Trung tâm", delta_color="off")
        c3.metric("🔺 Ước lượng cao", f"{high:.2f} tr", delta="+15%", delta_color="off")

        q33, q66 = df["price_million_vnd"].quantile(0.33), df["price_million_vnd"].quantile(0.66)
        if predicted_price < q33:
            segment, sc = "💵 Giá rẻ", "#10b981"
        elif predicted_price < q66:
            segment, sc = "💰 Tầm trung", "#f59e0b"
        else:
            segment, sc = "💎 Cao cấp", "#ef4444"

        if predict_clicked:
            history_item = {
                "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Nguồn nhập": input_source,
                "Hãng": specs["brand"],
                "RAM (GB)": specs["ram_gb"],
                "Bộ nhớ (GB)": specs["storage_gb"],
                "Pin (mAh)": specs["battery_mah"],
                "Camera (MP)": specs["camera_mp"],
                "Màn hình (inch)": specs["screen_inch"],
                "CPU cores": specs["cpu_cores"],
                "Giá dự đoán (triệu VND)": round(float(predicted_price), 2),
                "Ước lượng thấp": round(float(low), 2),
                "Ước lượng cao": round(float(high), 2),
                "Phân khúc": segment,
            }
            st.session_state.prediction_history.insert(0, history_item)
            st.session_state.prediction_history = st.session_state.prediction_history[:50]

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {sc}22, {sc}11);
                    padding: 1.25rem; border-radius: 12px; border-left: 5px solid {sc}; margin: 1rem 0;'>
            <p style='margin: 0; font-size: 1.1rem;'>
                <b>Phân khúc:</b> <span style='color: {sc}; font-size: 1.3rem;'>{segment}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Similar phones
        render_section_header("🔍", "Điện thoại tương tự")
        sim = df.copy()
        nf = ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "screen_inch", "cpu_cores"]
        for f in nf:
            sim[f + "_norm"] = (sim[f] - sim[f].min()) / (sim[f].max() - sim[f].min())
            qn = (specs[f] - df[f].min()) / (df[f].max() - df[f].min())
            sim[f + "_diff"] = (sim[f + "_norm"] - qn) ** 2
        sim["distance"] = sim[[f + "_diff" for f in nf]].sum(axis=1) ** 0.5
        sim["brand_match"] = (sim["brand"] == specs["brand"]).astype(int)
        sm = sim.sort_values(["brand_match", "distance"], ascending=[False, True]).head(5)
        sm_disp = sm[["brand", "model", "ram_gb", "storage_gb", "camera_mp",
                      "battery_mah", "price_million_vnd"]].rename(columns={
            "brand": "Hãng", "model": "Model", "ram_gb": "RAM",
            "storage_gb": "Bộ nhớ", "camera_mp": "Camera", "battery_mah": "Pin",
            "price_million_vnd": "Giá (tr)",
        })
        st.dataframe(sm_disp, use_container_width=True, hide_index=True)

        # Position on market
        render_section_header("📊", "Vị trí trên thị trường")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["price_million_vnd"], nbinsx=35,
                                    marker=dict(color=COLORS["primary"], opacity=0.5),
                                    name="Thị trường"))
        fig.add_vline(x=predicted_price, line=dict(color=COLORS["success"], width=4),
                      annotation_text=f"🎯 Dự đoán: {predicted_price:.2f} tr",
                      annotation_position="top")
        fig.add_vrect(x0=low, x1=high, fillcolor=COLORS["success"], opacity=0.15,
                      layer="below", line_width=0)
        fig.update_layout(height=400, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="Giá (triệu VNĐ)"),
                          yaxis=dict(gridcolor="#f1f5f9", title="Số điện thoại"),
                          showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        percentile = (df["price_million_vnd"] < predicted_price).mean() * 100
        render_info_box(f"📊 Giá dự đoán thuộc <b>top {100 - percentile:.1f}%</b> "
                        f"(cao hơn {percentile:.1f}% số điện thoại).", kind="info")

    render_section_header("🕘", "Lịch sử dự đoán")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.download_button(
                "⬇️ Tải lịch sử CSV",
                data=history_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"lich_su_du_doan_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("🗑️ Xóa lịch sử", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        render_info_box("ℹ️ Chưa có lịch sử. Bấm `Dự đoán giá` để lưu kết quả vào danh sách.", kind="info")


# ═══════════════════════════════════════════════════════════════
#                 6) HIỆU NĂNG MÔ HÌNH
# ═══════════════════════════════════════════════════════════════
elif PAGE == "🤖 Hiệu năng Mô hình":
    render_page_header(
        icon="🤖",
        title="Hiệu năng Mô hình",
        subtitle="So sánh các thuật toán Machine Learning đã huấn luyện",
    )

    model = load_model()
    if model is None:
        render_info_box(
            "⚠️ Chưa có mô hình. Chạy `python main.py` trước hoặc nhấn nút bên dưới để huấn luyện ngay.",
            kind="warning",
        )
        if st.button("🔧 Huấn luyện mô hình ngay", use_container_width=True):
            with st.spinner("⏳ Đang huấn luyện mô hình..."):
                from src.model_training import run_training

                try:
                    run_training()
                    st.success(
                        "✅ Đã huấn luyện xong! Mô hình đã được lưu vào `outputs/models/best_model.pkl`."
                    )
                    st.cache_resource.clear()
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"❌ Huấn luyện thất bại: {e}")
            st.experimental_rerun()
        st.stop()

    st.sidebar.markdown("### 🔧 Thao tác")
    if st.sidebar.button("🔄 Huấn luyện lại", use_container_width=True):
        with st.spinner("⏳ Đang huấn luyện..."):
            from src.model_training import run_training
            run_training()
            st.cache_resource.clear()
            st.cache_data.clear()
        st.sidebar.success("✅ Đã train xong!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**MAE** - Sai số tuyệt đối TB\n\n"
        "**RMSE** - Căn của MSE\n\n"
        "**R²** - Hệ số xác định (0-1, càng gần 1 càng tốt)"
    )

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        render_info_box("⚠️ Chạy `python -m src.model_training` trước.", kind="warning")
        st.stop()

    comparison = pd.read_csv(comparison_path)
    best_model_name = comparison.iloc[0]["Model"]

    render_section_header("🏆", "Mô hình tốt nhất")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🥇 Mô hình", best_model_name)
    c2.metric("🎯 R²", f"{comparison.iloc[0]['R²']:.4f}",
              delta=f"{comparison.iloc[0]['R²']*100:.1f}%", delta_color="off")
    c3.metric("📏 MAE", f"{comparison.iloc[0]['MAE']:.3f} tr")
    c4.metric("📐 RMSE", f"{comparison.iloc[0]['RMSE']:.3f} tr")

    render_section_header("📊", "Bảng so sánh chi tiết")
    disp = comparison.copy()
    for c in ["MAE", "RMSE"]:
        disp[c] = disp[c].round(3)
    for c in ["R²", "CV R² (mean)", "CV R² (std)"]:
        disp[c] = disp[c].round(4)
    disp.insert(0, "Hạng", ["🥇", "🥈", "🥉", "4", "5"][:len(disp)])
    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={
                     "R²": st.column_config.ProgressColumn("R² (Test)",
                         min_value=0, max_value=1, format="%.4f"),
                     "CV R² (mean)": st.column_config.ProgressColumn("CV R²",
                         min_value=0, max_value=1, format="%.4f"),
                     "MAE": st.column_config.NumberColumn("MAE (tr)", format="%.3f"),
                     "RMSE": st.column_config.NumberColumn("RMSE (tr)", format="%.3f"),
                 })

    render_section_header("📈", "Biểu đồ so sánh")
    c1, c2, c3 = st.columns(3)
    with c1:
        s = comparison.sort_values("MAE")
        fig = go.Figure(go.Bar(y=s["Model"], x=s["MAE"], orientation="h",
                                marker=dict(color=px.colors.sequential.Reds_r[:len(s)]),
                                text=[f"{v:.3f}" for v in s["MAE"]], textposition="outside"))
        fig.update_layout(title="MAE (thấp=tốt)", height=400, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9"), margin=dict(t=50, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        s = comparison.sort_values("RMSE")
        fig = go.Figure(go.Bar(y=s["Model"], x=s["RMSE"], orientation="h",
                                marker=dict(color=px.colors.sequential.Oranges_r[:len(s)]),
                                text=[f"{v:.3f}" for v in s["RMSE"]], textposition="outside"))
        fig.update_layout(title="RMSE (thấp=tốt)", height=400, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9"), margin=dict(t=50, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        s = comparison.sort_values("R²", ascending=False)
        fig = go.Figure(go.Bar(y=s["Model"], x=s["R²"], orientation="h",
                                marker=dict(color=px.colors.sequential.Greens[-len(s):][::-1]),
                                text=[f"{v:.4f}" for v in s["R²"]], textposition="outside"))
        fig.update_layout(title="R² (cao=tốt)", height=400, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", range=[0, 1]),
                          margin=dict(t=50, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # Predictions vs actual
    render_section_header("🎯", f"Dự đoán vs Thực tế - {best_model_name}")
    X = df[["ram_gb", "storage_gb", "battery_mah", "camera_mp",
            "screen_inch", "cpu_cores", "brand"]]
    y = df["price_million_vnd"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📍 Dự đoán vs Thực tế")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
            marker=dict(color=residuals, colorscale="RdYlGn_r",
                        size=6, opacity=0.6, line=dict(width=0.5, color="white"),
                        colorbar=dict(title="Phần dư"), cmid=0),
            name="Điện thoại"))
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                  line=dict(color="red", dash="dash", width=2),
                                  name="Hoàn hảo"))
        fig.update_layout(height=450, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="Thực tế (triệu VND)"),
                          yaxis=dict(gridcolor="#f1f5f9", title="Dự đoán (triệu VND)"),
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### 📊 Phân bố phần dư")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                    marker=dict(color=COLORS["primary"])))
        fig.add_vline(x=0, line=dict(color=COLORS["danger"], dash="dash", width=2),
                      annotation_text="Không sai số")
        fig.update_layout(height=450, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="Phần dư"),
                          yaxis=dict(gridcolor="#f1f5f9", title="Số lượng"),
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"- **TB:** {residuals.mean():.3f} tr\n"
                    f"- **Std:** {residuals.std():.3f} tr\n"
                    f"- **|Sai số| TB:** {residuals.abs().mean():.3f} tr")

    # Feature importance
    render_section_header("⭐", "Tầm quan trọng đặc trưng")
    estimator = model.named_steps.get("model", None)

    if estimator is not None and hasattr(estimator, "feature_importances_"):
        pre = model.named_steps["pre"]
        num_features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                        "screen_inch", "cpu_cores"]
        cat_features = pre.named_transformers_["cat"].get_feature_names_out(["brand"]).tolist()
        imp_df = pd.DataFrame({
            "feature": num_features + cat_features,
            "importance": estimator.feature_importances_,
        })
        imp_df["group"] = imp_df["feature"].apply(
            lambda x: "brand" if x.startswith("brand_") else x)
        gr = imp_df.groupby("group")["importance"].sum().sort_values(ascending=True)
        label_map = {"ram_gb": "🧠 RAM", "storage_gb": "💾 Bộ nhớ",
                     "battery_mah": "🔋 Pin", "camera_mp": "📸 Camera",
                     "screen_inch": "📱 Màn hình", "cpu_cores": "⚙️ CPU", "brand": "🏷️ Hãng"}
        gr.index = [label_map.get(f, f) for f in gr.index]

        fig = go.Figure(go.Bar(y=gr.index, x=gr.values, orientation="h",
            marker=dict(color=gr.values, colorscale="Viridis", showscale=False),
            text=[f"{v:.3f}" for v in gr.values], textposition="outside"))
        fig.update_layout(height=400, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="Mức độ quan trọng"),
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        render_info_box(f"⭐ <b>{gr.idxmax()}</b> là yếu tố quan trọng nhất "
                        f"(trọng số {gr.max():.3f})", kind="info")
    elif estimator is not None and hasattr(estimator, "coef_"):
        render_info_box(f"ℹ️ Mô hình <b>{best_model_name}</b> dùng hệ số tuyến tính "
                        "(coefficients) thay cho feature_importances_.", kind="info")
        pre = model.named_steps["pre"]
        num_features = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                        "screen_inch", "cpu_cores"]
        cat_features = pre.named_transformers_["cat"].get_feature_names_out(["brand"]).tolist()
        cdf = pd.DataFrame({"feature": num_features + cat_features, "coef": estimator.coef_})
        cdf["abs_coef"] = cdf["coef"].abs()
        top = cdf.sort_values("abs_coef", ascending=True).tail(15)

        fig = go.Figure(go.Bar(y=top["feature"], x=top["coef"], orientation="h",
            marker=dict(color=top["coef"], colorscale="RdYlGn", cmid=0),
            text=[f"{v:+.2f}" for v in top["coef"]], textposition="outside"))
        fig.update_layout(title="Hệ số (top 15)", height=500, plot_bgcolor="white",
                          xaxis=dict(gridcolor="#f1f5f9", title="Hệ số"),
                          margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📚 Giải thích các thuật toán"):
        st.markdown("""
        - **Linear Regression** — Mô hình tuyến tính đơn giản, dễ hiểu, chạy nhanh
        - **Ridge Regression** — Linear + L2 regularization, giảm overfitting
        - **Decision Tree** — Chia dữ liệu theo cây nhị phân, dễ interpret
        - **Random Forest** — Ensemble nhiều tree, bền vững, ít overfit
        - **Gradient Boosting** — Ensemble tuần tự, accuracy cao nhưng khó tune
        """)


# ═══════════════════════════════════════════════════════════════
#                      7) GIỚI THIỆU
# ═══════════════════════════════════════════════════════════════
elif PAGE == "ℹ️  Giới thiệu":
    render_page_header(
        icon="ℹ️",
        title="Giới thiệu Project",
        subtitle="Mobile Price Analytics - Web app phân tích & dự đoán giá điện thoại",
    )

    render_section_header("🎯", "Mục đích")
    st.markdown("""
    Project này giúp người dùng:
    - 📊 **Phân tích** toàn cảnh thị trường điện thoại
    - 🔍 **Khám phá** các yếu tố ảnh hưởng đến giá
    - 🔮 **Dự đoán** giá của điện thoại mới dựa trên cấu hình
    - 📈 **Trực quan hóa** dữ liệu qua nhiều biểu đồ tương tác
    """)

    render_section_header("🛠️", "Công nghệ sử dụng")
    c1, c2, c3 = st.columns(3)
    techs = [
        ("🐍 Python 3.10+", "Ngôn ngữ chính"),
        ("🌐 Streamlit", "Framework web"),
        ("📊 Plotly", "Biểu đồ tương tác"),
        ("🐼 Pandas", "Xử lý dữ liệu"),
        ("🔢 NumPy", "Tính toán số"),
        ("🤖 scikit-learn", "Machine Learning"),
        ("📉 Matplotlib", "Biểu đồ tĩnh"),
        ("🎨 Seaborn", "Thống kê"),
        ("💾 Joblib", "Lưu mô hình"),
    ]
    for i, (n, d) in enumerate(techs):
        with [c1, c2, c3][i % 3]:
            st.markdown(f"""
            <div style='background:white;padding:1rem;border-radius:10px;
                        border:1px solid #e2e8f0;margin-bottom:0.75rem;'>
                <div style='font-size:1.1rem;font-weight:600;color:#0f172a;'>{n}</div>
                <div style='color:#64748b;font-size:0.9rem;'>{d}</div>
            </div>
            """, unsafe_allow_html=True)

    render_section_header("📄", "Các trang trong app")
    pages_info = [
        ("🏠", "Dashboard", "Tổng quan KPI & biểu đồ tóm tắt"),
        ("📊", "Khám phá Dữ liệu", "Bảng lọc, sắp xếp, tải CSV"),
        ("📈", "Phân tích & Insight", "EDA 5 góc nhìn chuyên sâu"),
        ("🎨", "Biểu đồ Trực quan", "9 loại biểu đồ Plotly tương tác"),
        ("🔮", "Dự đoán Giá", "Form nhập cấu hình → dự đoán ML"),
        ("🤖", "Hiệu năng Mô hình", "So sánh MAE/RMSE/R² 5 thuật toán"),
        ("ℹ️", "Giới thiệu", "Tài liệu & hướng dẫn"),
    ]
    for icon, title, desc in pages_info:
        st.markdown(f"""
        <div style='background:white;padding:1rem 1.25rem;border-radius:10px;
                    border:1px solid #e2e8f0;margin-bottom:0.75rem;
                    display:flex;align-items:center;gap:1rem;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div>
                <div style='font-weight:700;font-size:1.05rem;color:#0f172a;'>{title}</div>
                <div style='color:#64748b;font-size:0.9rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_section_header("🔄", "Pipeline xử lý")
    st.markdown("""
    ```
    📥 Dữ liệu gốc
        ↓
    🧹 Làm sạch (chuẩn hóa đơn vị → điền missing → loại outlier → dedupe)
        ↓
    📊 EDA (thống kê, hãng, cấu hình, tương quan)
        ↓
    🤖 Train (5 mô hình + CV 5-fold, chọn theo R²)
        ↓
    🔮 Deploy cho dự đoán
    ```
    """)

    render_section_header("📐", "Chỉ số đánh giá")
    st.markdown("""
    | Chỉ số | Ý nghĩa | Tốt khi |
    |--------|---------|---------|
    | **MAE** | Sai số tuyệt đối TB (triệu VNĐ) | Càng thấp càng tốt |
    | **RMSE** | Căn của MSE, phạt lỗi lớn | Càng thấp càng tốt |
    | **R²** | % biến thiên được giải thích | Càng gần 1 càng tốt |
    """)

    render_section_header("📚", "Hướng dẫn sử dụng")
    with st.expander("🚀 Cài đặt và chạy"):
        st.code("""
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Sinh dữ liệu + train model (lần đầu)
python main.py

# 3. Khởi động web app
streamlit run app.py
""", language="bash")

    with st.expander("🔧 Dùng dữ liệu thật"):
        st.markdown("""
        Thay `data/mobile_phones.csv` bằng file của bạn với các cột:
        `brand, model, ram_gb, storage_gb, battery_mah, camera_mp, screen_inch, cpu_cores, price_million_vnd`

        Sau đó chạy lại:
        ```bash
        python -m src.data_cleaning
        python -m src.model_training
        ```
        Rồi restart web app.
        """)

    with st.expander("🐛 Lỗi thường gặp"):
        st.markdown("""
        - **"Không tìm thấy dữ liệu sạch"** → Chạy `python main.py` trước
        - **"Chưa có mô hình"** → Chạy `python -m src.model_training`
        - **Port 8501 bị chiếm** → `streamlit run app.py --server.port 8502`
        """)

    st.markdown("---")
    st.markdown("""

    """, unsafe_allow_html=True)
