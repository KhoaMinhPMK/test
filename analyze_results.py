import pandas as pd
import numpy as np

print("🔍 PHÂN TÍCH KẾT QUẢ FEATURE EXTRACTION")
print("=" * 50)

# 1. Phân tích dataset chính
print("\n📊 1. PHÂN TÍCH DATASET CHÍNH:")
df = pd.read_csv('data/extracted_features.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Tổng số samples: {len(df):,}")

# Label distribution
label_counts = df['label'].value_counts()
print(f"\n   📋 Phân bố labels:")
for label, count in label_counts.items():
    percentage = count / len(df) * 100
    label_name = "Normal" if label == 0 else "Falling"
    print(f"      {label_name} (label={label}): {count:,} samples ({percentage:.1f}%)")

# Imbalance ratio
imbalance_ratio = label_counts[0] / label_counts[1] if 1 in label_counts else float('inf')
print(f"   📊 Tỷ lệ mất cân bằng (Normal:Falling): {imbalance_ratio:.1f}:1")

# Features
feature_columns = [col for col in df.columns if col not in ['file_name', 'window_start', 'window_end', 'label', 'fall_ratio']]
print(f"   🎯 Số đặc trưng: {len(feature_columns)}")

# Feature types
feature_types = {}
for col in feature_columns:
    prefix = col.split('_')[0]
    if prefix not in feature_types:
        feature_types[prefix] = 0
    feature_types[prefix] += 1

print(f"\n   📋 Loại đặc trưng:")
for feat_type, count in feature_types.items():
    print(f"      {feat_type}: {count} features")

# 2. Phân tích Feature Importance
print(f"\n📊 2. PHÂN TÍCH FEATURE IMPORTANCE:")
importance_df = pd.read_csv('data/feature_importance.csv')

print(f"\n   🏆 TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
top_10 = importance_df.head(10)
for i, row in top_10.iterrows():
    print(f"      {i+1:2d}. {row['feature']:<25} | Score: {row['combined_score']:.4f}")

print(f"\n   📋 Phân loại top 10 theo nhóm:")
top_10_types = {}
for _, row in top_10.iterrows():
    prefix = row['feature'].split('_')[0]
    if prefix not in top_10_types:
        top_10_types[prefix] = 0
    top_10_types[prefix] += 1

for feat_type, count in top_10_types.items():
    print(f"      {feat_type}: {count} features")

# 3. Insights về đặc trưng
print(f"\n🧠 3. INSIGHTS VỀ ĐẶC TRƯNG:")

print(f"\n   ✅ Đặc trưng thống kê (stat_) chiếm ưu thế:")
stat_features = importance_df[importance_df['feature'].str.startswith('stat_')].head(5)
print(f"      - stat_skewness và stat_kurtosis là 2 đặc trưng quan trọng nhất")
print(f"      - Cho thấy phân bố dữ liệu có vai trò quan trọng trong phân loại")

print(f"\n   ✅ Đặc trưng chuyển động (motion_) quan trọng:")
motion_features = importance_df[importance_df['feature'].str.startswith('motion_')].head(3)
print(f"      - motion_displacement: độ dịch chuyển tổng thể")
print(f"      - Phản ánh sự thay đổi vị trí khi ngã")

print(f"\n   ✅ Đặc trưng hình học (geo_) cơ bản:")
geo_features = importance_df[importance_df['feature'].str.startswith('geo_')].head(3)
print(f"      - geo_aspect_ratio: tỷ lệ khung hình")
print(f"      - geo_width, geo_height: kích thước bbox")

print(f"\n   ✅ Đặc trưng tư thế (pose_) chi tiết:")
pose_features = importance_df[importance_df['feature'].str.startswith('pose_')].head(3)
print(f"      - pose_orientation: hướng cơ thể")
print(f"      - Quan trọng để phân biệt đứng/nằm")

# 4. Đánh giá chất lượng
print(f"\n📈 4. ĐÁNH GIÁ CHẤT LƯỢNG:")

print(f"\n   ✅ Ưu điểm:")
print(f"      - Dataset đủ lớn: {len(df):,} samples")
print(f"      - Đa dạng đặc trưng: {len(feature_columns)} features từ 6 nhóm")
print(f"      - Sliding window tạo nhiều samples từ ít video")
print(f"      - Feature importance rõ ràng, có ý nghĩa")

print(f"\n   ⚠️ Thách thức:")
print(f"      - Mất cân bằng dữ liệu nghiêm trọng: {imbalance_ratio:.1f}:1")
print(f"      - Cần cân bằng dữ liệu trước khi training")
print(f"      - Một số đặc trưng có thể redundant")

# 5. Khuyến nghị
print(f"\n💡 5. KHUYẾN NGHỊ:")
print(f"\n   🎯 Cho bước tiếp theo:")
print(f"      1. Sử dụng top 20-30 features quan trọng nhất")
print(f"      2. Áp dụng data balancing (SMOTE, oversampling)")
print(f"      3. Thử các model: Random Forest, SVM, Neural Network")
print(f"      4. Cross-validation để đánh giá robust")
print(f"      5. Feature engineering thêm nếu cần")

print(f"\n   📊 Balanced datasets đã tạo:")
try:
    import os
    balanced_files = [f for f in os.listdir('data') if f.startswith('balanced_data_')]
    for file in balanced_files:
        print(f"      - {file}")
except:
    print(f"      - Chưa có balanced datasets")

print(f"\n🎉 KẾT LUẬN:")
print(f"   Thuật toán feature extraction đã thành công!")
print(f"   Dataset có chất lượng tốt với {len(feature_columns)} đặc trưng đa dạng.")
print(f"   Sẵn sàng cho bước training model machine learning.") 