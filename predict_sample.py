import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sample prediction using trained model.")
    parser.add_argument("--model-path", type=str, default="artifacts/mobile_price_model.pkl")
    parser.add_argument(
        "--input-json",
        type=str,
        default="sample_input.json",
        help="Path to JSON object or array of objects for prediction.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_json)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = joblib.load(model_path)

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = [payload]

    df = pd.DataFrame(payload)
    preds = model.predict(df)

    print("Predictions:")
    for idx, price in enumerate(preds, start=1):
        print(f"Sample {idx}: {price:.2f}")


if __name__ == "__main__":
    main()
