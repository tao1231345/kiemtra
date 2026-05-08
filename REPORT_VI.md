# BAO CAO DO AN: PHAN TICH VA DU DOAN GIA DIEN THOAI

## 1) Muc tieu de tai

De tai xay dung he thong du doan gia dien thoai dua tren cac thong so ky thuat va thuong hieu san pham.  
Mo hinh hoc may duoc huan luyen tren tap du lieu lon lay tu Kaggle thong qua API de dam bao tinh tu dong va kha nang mo rong.

## 2) Nguon du lieu lon

- Nguon: Kaggle dataset (tai tu dong bang `kaggle datasets download`).
- Vi du dataset da dung: `PromptCloudHQ/flipkart-products`.
- Dac diem:
  - So luong mau lon (du lieu thuong mai dien tu voi nhieu san pham).
  - Co cot gia va thong tin ky thuat/co ban cua san pham.

## 3) Quy trinh xu ly

1. Tai du lieu tu dong bang Kaggle API.
2. Doc file CSV va tu dong chon file phu hop nhat cho bai toan mobile price.
3. Tu dong tim cot muc tieu gia (price/cost/mrp/selling...).
4. Lam sach du lieu:
   - Chuan hoa gia tu chuoi ve so.
   - Bo cot khong can thiet (url, link, image, id...).
   - Xu ly gia tri thieu.
5. Ma hoa dac trung:
   - So: median imputation.
   - Chuoi: most_frequent + one-hot encoding.
6. Chia tap train/test (80/20).
7. Huan luyen 2 mo hinh:
   - LinearRegression (baseline)
   - RandomForestRegressor (mo hinh chinh)
8. Danh gia ket qua bang MAE, RMSE, R2.
9. Luu mo hinh va metrics de tai su dung.

## 4) Cong cu va cong nghe

- Python
- pandas, numpy
- scikit-learn
- Kaggle API
- joblib

## 5) Ket qua

Sau khi chay script:

```bash
python auto_mobile_price_pipeline.py --dataset "PromptCloudHQ/flipkart-products"
```

He thong tao:

- `artifacts/mobile_price_model.pkl`
- `artifacts/metrics.json`

Trich cac metric tu `metrics.json` va dien vao bang:

- LinearRegression: MAE = ..., RMSE = ..., R2 = ...
- RandomForestRegressor: MAE = ..., RMSE = ..., R2 = ...

Mo hinh tot nhat duoc chon theo R2 cao hon (dong thoi MAE/RMSE thap hon).

## 6) Ket luan

De tai da xay dung thanh cong quy trinh du doan gia dien thoai tu dong tu thu thap du lieu den huan luyen va danh gia mo hinh.  
Huong phat trien:

- Thu nghiem XGBoost/LightGBM de tang do chinh xac.
- Them tinh nang giai thich mo hinh (feature importance/SHAP).
- Trien khai API web de du doan theo thoi gian thuc.
