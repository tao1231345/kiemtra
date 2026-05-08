import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_DATASET = "PromptCloudHQ/flipkart-products"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "mobile_price_model.pkl"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
DEFAULT_ANALYSIS_PATH = DEFAULT_OUTPUT_DIR / "analysis_summary.json"
DEFAULT_PREDICTIONS_PATH = DEFAULT_OUTPUT_DIR / "test_predictions.csv"


def run_command(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )


def ensure_kaggle_installed() -> None:
    try:
        import kaggle  # noqa: F401
    except Exception:
        run_command([sys.executable, "-m", "pip", "install", "kaggle"])


def ensure_kaggle_credentials() -> None:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return
    raise FileNotFoundError(
        "Kaggle API key not found. Please place kaggle.json at "
        f"{kaggle_json} (from Kaggle Account -> API -> Create New Token)."
    )


def download_dataset(dataset: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    run_command([sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(data_dir), "--unzip"])


def list_csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.csv"))


def choose_best_csv(csv_files: list[Path]) -> Path:
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory after download.")
    best_file = None
    best_score = -1
    for path in csv_files:
        score = 0
        lower = path.name.lower()
        if "mobile" in lower or "phone" in lower or "smart" in lower:
            score += 5
        if "price" in lower:
            score += 3
        size = path.stat().st_size
        score += min(size // 1_000_000, 10)
        if score > best_score:
            best_score = score
            best_file = path
    if best_file is None:
        raise FileNotFoundError("Could not choose a CSV file for training.")
    return best_file


def normalize_price_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def find_target_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if "price" in lower or "cost" in lower or "mrp" in lower or "selling" in lower:
            candidates.append(col)
    if not candidates:
        raise ValueError(f"No price-like target column found. Columns: {list(df.columns)}")

    best_col = None
    best_valid = -1
    for col in candidates:
        values = normalize_price_column(df[col])
        valid_count = values.notna().sum()
        if valid_count > best_valid:
            best_valid = valid_count
            best_col = col

    if best_col is None:
        raise ValueError("Could not detect usable target column.")
    return best_col


def clean_feature_columns(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = normalize_price_column(df[target_col])
    X = df.drop(columns=[target_col]).copy()

    drop_cols = []
    for col in X.columns:
        lower = col.lower()
        if any(token in lower for token in ["url", "link", "image", "id"]):
            drop_cols.append(col)
    if drop_cols:
        X = X.drop(columns=drop_cols)

    X = X.loc[y.notna()].copy()
    y = y.loc[y.notna()].copy()

    # Remove very-high-cardinality columns that mostly hurt baseline models.
    high_card_cols = []
    for col in X.select_dtypes(include=["object"]).columns:
        nunique = X[col].nunique(dropna=True)
        if nunique > max(200, len(X) * 0.5):
            high_card_cols.append(col)
    if high_card_cols:
        X = X.drop(columns=high_card_cols)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_analysis_report(
    raw_df: pd.DataFrame,
    cleaned_X: pd.DataFrame,
    cleaned_y: pd.Series,
    target_col: str,
    output_dir: Path,
) -> None:
    summary = {
        "raw_rows": int(len(raw_df)),
        "raw_cols": int(len(raw_df.columns)),
        "clean_rows": int(len(cleaned_X)),
        "clean_cols": int(len(cleaned_X.columns)),
        "target_column": target_col,
        "target_stats": {
            "min": float(cleaned_y.min()),
            "max": float(cleaned_y.max()),
            "mean": float(cleaned_y.mean()),
            "median": float(cleaned_y.median()),
        },
        "missing_values_top10": {
            k: int(v)
            for k, v in raw_df.isna().sum().sort_values(ascending=False).head(10).items()
        },
        "feature_types": {
            "numeric_count": int(
                len(cleaned_X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns)
            ),
            "categorical_count": int(
                len(cleaned_X.select_dtypes(include=["object", "category", "bool"]).columns)
            ),
        },
    }
    with open(output_dir / DEFAULT_ANALYSIS_PATH.name, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save a quick histogram for report/slides.
    plt.figure(figsize=(8, 5))
    plt.hist(cleaned_y, bins=40, edgecolor="black")
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png", dpi=120)
    plt.close()


def train_models(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, list[dict], pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X)

    linear = Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression())])
    forest = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )

    linear.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    linear_metrics = evaluate_model("LinearRegression", y_test, linear.predict(X_test))
    forest_pred = forest.predict(X_test)
    forest_metrics = evaluate_model("RandomForestRegressor", y_test, forest_pred)
    metrics = [linear_metrics, forest_metrics]
    metrics.sort(key=lambda item: item["r2"], reverse=True)

    predictions_df = X_test.copy()
    predictions_df["actual_price"] = y_test.values
    predictions_df["predicted_price"] = forest_pred
    predictions_df["abs_error"] = np.abs(predictions_df["actual_price"] - predictions_df["predicted_price"])

    best = max([(linear, linear_metrics), (forest, forest_metrics)], key=lambda pair: pair[1]["r2"])
    return best[0], metrics, predictions_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto download + analyze + train mobile price prediction pipeline.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Kaggle dataset in owner/name format.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download and use local CSV.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ensure_kaggle_installed()
        ensure_kaggle_credentials()
        print(f"Downloading dataset: {args.dataset}")
        download_dataset(args.dataset, data_dir)

    csv_files = list_csv_files(data_dir)
    csv_path = choose_best_csv(csv_files)
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded rows={len(df)}, cols={len(df.columns)}")

    target_col = find_target_column(df)
    print(f"Detected target column: {target_col}")
    X, y = clean_feature_columns(df, target_col)
    print(f"Training rows after cleaning: {len(X)}")

    if len(X) < 500:
        print("Warning: dataset size is small for a 'big data' claim. Consider another larger Kaggle dataset.")

    save_analysis_report(df, X, y, target_col, output_dir)
    model, metrics, predictions_df = train_models(X, y)
    joblib.dump(model, output_dir / DEFAULT_MODEL_PATH.name)
    with open(output_dir / DEFAULT_METRICS_PATH.name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    predictions_df.to_csv(output_dir / DEFAULT_PREDICTIONS_PATH.name, index=False)

    print("\nEvaluation metrics:")
    for item in metrics:
        print(
            f"{item['model']}: MAE={item['mae']:.2f}, "
            f"RMSE={item['rmse']:.2f}, R2={item['r2']:.4f}"
        )
    print(f"\nSaved model: {output_dir / DEFAULT_MODEL_PATH.name}")
    print(f"Saved metrics: {output_dir / DEFAULT_METRICS_PATH.name}")
    print(f"Saved analysis summary: {output_dir / DEFAULT_ANALYSIS_PATH.name}")
    print(f"Saved test predictions: {output_dir / DEFAULT_PREDICTIONS_PATH.name}")
    print(f"Saved chart: {output_dir / 'price_distribution.png'}")


if __name__ == "__main__":
    main()
