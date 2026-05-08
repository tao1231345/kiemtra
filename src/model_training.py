"""
Module huấn luyện và đánh giá các mô hình dự đoán giá điện thoại.

So sánh 5 mô hình:
1. Linear Regression
2. Ridge Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting

Chỉ số đánh giá:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42

# Định nghĩa các đặc trưng
NUMERIC_FEATURES = ["ram_gb", "storage_gb", "battery_mah", "camera_mp",
                    "screen_inch", "cpu_cores"]
CATEGORICAL_FEATURES = ["brand"]
TARGET = "price_million_vnd"


def prepare_data(df: pd.DataFrame):
    """Chia dữ liệu train/test và build preprocessor."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )

    # ColumnTransformer: scale số + one-hot encode categorical
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    print(f"📦 Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")
    print(f"📐 Features: {len(NUMERIC_FEATURES)} số + {len(CATEGORICAL_FEATURES)} categorical")

    return X_train, X_test, y_train, y_test, preprocessor


def get_models() -> dict:
    """Định nghĩa các mô hình cần so sánh."""
    return {
        "Linear Regression":    LinearRegression(),
        "Ridge Regression":     Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree":        DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Tính MAE, RMSE, R² trên tập test."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R²": r2, "y_pred": y_pred}


def train_and_compare(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Huấn luyện tất cả mô hình và trả về bảng so sánh."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    results = []
    trained_models = {}

    print("\n" + "─" * 70)
    print(f"{'Mo hinh':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'CV R²':>12}")
    print("─" * 70)

    for name, estimator in get_models().items():
        # Pipeline = preprocessor + model
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", estimator),
        ])

        # Fit
        pipeline.fit(X_train, y_train)

        # Đánh giá trên test
        metrics = evaluate_model(pipeline, X_test, y_test)

        # Cross-validation 5 folds để kiểm tra ổn định
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1,
        )
        cv_mean = cv_scores.mean()

        results.append({
            "Model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R²": metrics["R²"],
            "CV R² (mean)": cv_mean,
            "CV R² (std)": cv_scores.std(),
        })
        trained_models[name] = pipeline

        print(f"{name:<22} {metrics['MAE']:>10.3f} {metrics['RMSE']:>10.3f} "
              f"{metrics['R²']:>10.4f} {cv_mean:>8.4f} ± {cv_scores.std():.3f}")

    print("─" * 70)

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)

    # Xác định mô hình tốt nhất
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_name}")
    print(f"   → R² = {results_df.iloc[0]['R²']:.4f} "
          f"(giải thích được {results_df.iloc[0]['R²']*100:.1f}% biến thiên của giá)")
    print(f"   → MAE = {results_df.iloc[0]['MAE']:.3f} triệu VND "
          f"(sai số trung bình)")

    # Lưu mô hình tốt nhất
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"💾 Đã lưu mô hình tốt nhất tại: {model_path}")

    # Lưu báo cáo so sánh
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"📝 Đã lưu bảng so sánh tại: {report_path}")

    # Vẽ biểu đồ so sánh
    plot_model_comparison(results_df, output_dir / "figures")
    plot_predictions_vs_actual(best_model, X_test, y_test, best_name, output_dir / "figures")
    plot_feature_importance(best_model, preprocessor, best_name, output_dir / "figures")

    return results_df, best_model


def plot_model_comparison(results_df: pd.DataFrame, figures_dir: Path) -> None:
    """Biểu đồ so sánh 3 chỉ số giữa các mô hình."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("MAE", "MAE (thap = tot)", "Reds_r"),
               ("RMSE", "RMSE (thap = tot)", "Oranges_r"),
               ("R²", "R² (cao = tot)", "Greens")]

    for ax, (col, title, cmap) in zip(axes, metrics):
        df_sorted = results_df.sort_values(col, ascending=(col != "R²"))
        colors = sns.color_palette(cmap, n_colors=len(df_sorted))
        bars = ax.barh(df_sorted["Model"], df_sorted[col], color=colors, edgecolor="black")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(col)
        for bar, val in zip(bars, df_sorted[col]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:.3f}", va="center", fontsize=10)

    plt.suptitle("So sanh hieu nang cac mo hinh", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = figures_dir / "06_model_comparison.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_predictions_vs_actual(model, X_test, y_test, model_name: str,
                                figures_dir: Path) -> None:
    """Biểu đồ: Giá dự đoán vs Giá thực tế."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter dự đoán vs thực tế
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolor="white")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=2, label="Du doan hoan hao")
    ax1.set_xlabel("Gia thuc te (trieu VND)")
    ax1.set_ylabel("Gia du doan (trieu VND)")
    ax1.set_title(f"Du doan vs Thuc te - {model_name}", fontsize=13, fontweight="bold")
    ax1.legend()

    # Residuals
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.5, color="purple", edgecolor="white")
    ax2.axhline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Gia du doan (trieu VND)")
    ax2.set_ylabel("Phan du (Residuals)")
    ax2.set_title("Phan tich phan du", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = figures_dir / "07_predictions_vs_actual.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def plot_feature_importance(model, preprocessor, model_name: str,
                            figures_dir: Path) -> None:
    """Vẽ feature importance nếu mô hình hỗ trợ."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print(f"   ℹ  {model_name} không hỗ trợ feature_importances_, bỏ qua biểu đồ này")
        return

    # Lấy tên đặc trưng sau khi qua preprocessor
    pre = model.named_steps["pre"]
    num_names = NUMERIC_FEATURES
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = num_names + cat_names

    importances = estimator.feature_importances_
    # Gộp one-hot encoded về từng nhóm để dễ đọc
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp["group"] = df_imp["feature"].apply(
        lambda x: "brand" if x.startswith("brand_") else x
    )
    grouped = df_imp.groupby("group")["importance"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", n_colors=len(grouped))
    bars = ax.barh(grouped.index, grouped.values, color=colors, edgecolor="black")
    ax.set_xlabel("Muc do quan trong")
    ax.set_title(f"Feature Importance - {model_name}", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    path = figures_dir / "08_feature_importance.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"   ✅ Đã lưu: {path.name}")


def predict_new_phone(model_path: str, phone_specs: dict) -> float:
    """
    Dự đoán giá cho 1 điện thoại mới.

    Ví dụ:
        predict_new_phone("outputs/models/best_model.pkl", {
            "brand": "Samsung",
            "ram_gb": 8,
            "storage_gb": 256,
            "battery_mah": 5000,
            "camera_mp": 108,
            "screen_inch": 6.5,
            "cpu_cores": 8,
        })
    """
    model = joblib.load(model_path)
    df_input = pd.DataFrame([phone_specs])
    predicted_price = model.predict(df_input)[0]
    return predicted_price


def run_training(data_path: str = "data/mobile_phones_clean.csv",
                 output_dir: str = "outputs") -> pd.DataFrame:
    """Chạy toàn bộ pipeline huấn luyện."""
    print("\n" + "=" * 60)
    print("🤖 HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    df = pd.read_csv(data_path)
    results_df, best_model = train_and_compare(df, Path(output_dir))

    # Demo: dự đoán cho 1 cấu hình giả định
    print("\n" + "=" * 60)
    print("🎯 DEMO: Dự đoán giá cho 1 điện thoại mới")
    print("=" * 60)
    demo_phone = {
        "brand": "Samsung",
        "ram_gb": 8,
        "storage_gb": 256,
        "battery_mah": 5000,
        "camera_mp": 108,
        "screen_inch": 6.5,
        "cpu_cores": 8,
    }
    pred = best_model.predict(pd.DataFrame([demo_phone]))[0]
    print(f"   Cấu hình: {demo_phone}")
    print(f"   ➡  Giá dự đoán: {pred:.2f} triệu VND")

    return results_df


if __name__ == "__main__":
    run_training()