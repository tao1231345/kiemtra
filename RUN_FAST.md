# Fast run guide

Run all steps automatically (download Kaggle data, analysis, train model, save predictions):

```bash
pip install -r requirements.txt
python auto_mobile_price_pipeline.py --dataset "PromptCloudHQ/flipkart-products"
```

Output files:

- `artifacts/mobile_price_model.pkl`
- `artifacts/metrics.json`
- `artifacts/analysis_summary.json`
- `artifacts/test_predictions.csv`
- `artifacts/price_distribution.png`

If you already downloaded CSV files in `data/`, skip download:

```bash
python auto_mobile_price_pipeline.py --skip-download
```

Run a sample prediction after training:

```bash
python predict_sample.py --input-json sample_input.json
```

Notes:

- Place Kaggle API key at `C:\Users\<your-user>\.kaggle\kaggle.json`
- You can change dataset by replacing `--dataset "owner/name"`
