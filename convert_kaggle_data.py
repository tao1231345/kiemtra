import pandas as pd
import numpy as np

# Đọc file gốc
df = pd.read_csv('data/mobile_phones.csv')

# Đổi tên cột
df = df.rename(columns={
    'Brand': 'brand',
    'Model': 'model',
    'Storage ': 'storage_gb',
    'RAM ': 'ram_gb',
    'Screen Size (inches)': 'screen_inch',
    'Camera (MP)': 'camera_mp',
    'Battery Capacity (mAh)': 'battery_mah',
    'Price ($)': 'price_usd'
})

# Làm sạch dữ liệu
df['storage_gb'] = df['storage_gb'].str.replace('GB', '').str.strip().astype(int)
df['ram_gb'] = df['ram_gb'].str.replace('GB', '').str.strip().astype(int)
df['camera_mp'] = df['camera_mp'].astype(str)  # Giữ string vì có nhiều camera

# Chuyển giá USD sang triệu VND (1 USD ≈ 23,000 VND)
df['price_usd'] = df['price_usd'].str.replace('$', '').str.replace(',', '').str.strip().astype(float)
df['price_million_vnd'] = (df['price_usd'] * 23000) / 1000000

# Thêm cpu_cores ngẫu nhiên
np.random.seed(42)
df['cpu_cores'] = np.random.randint(4, 9, size=len(df))

# Sắp xếp cột theo yêu cầu
df = df[['brand', 'model', 'ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_inch', 'cpu_cores', 'price_million_vnd']]

# Lưu lại
df.to_csv('data/mobile_phones.csv', index=False)

print(f"Đã xử lý {len(df)} dòng dữ liệu từ Kaggle.")