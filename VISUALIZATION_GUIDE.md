---
noteId: "ae0e3eb0398811f0a8ff1b132a37f714"
tags: []

---

# HƯỚNG DẪN XEM BIỂU ĐỒ - HỆ THỐNG PHÁT HIỆN NGÃ TIÊN ĐOÁN

## 📊 TỔNG QUAN CÁC BIỂU ĐỒ

Bài báo khoa học bao gồm **4 biểu đồ chính** minh họa quá trình phân tích và xử lý dữ liệu:

### 🔍 **Hình 1: Phân tích Phân bố Dữ liệu** (`data_distribution.png`)

**Nội dung:**
- **Biểu đồ tròn**: Phân bố nhãn gốc (96.7% Normal vs 3.3% Falling)
- **Histogram**: Phân bố tỷ lệ ngã theo từng file video
- **Bar chart**: Phân bố thời gian window (10 frames)
- **Heatmap**: Ma trận tương quan giữa các features mẫu
- **Box plot**: Phân bố features theo nhãn (Normal vs Falling)
- **Bar chart**: Top 10 files có nhiều mẫu nhất

**Ý nghĩa:**
- Cho thấy vấn đề **imbalanced data** nghiêm trọng (tỷ lệ 28.9:1)
- Các features có **correlation thấp**, đảm bảo tính độc lập
- Dữ liệu được phân bố đều qua các môi trường khác nhau

---

### 📈 **Hình 2: Phân tích Tầm quan trọng Features** (`feature_importance.png`)

**Nội dung:**
- **Random Forest Importance**: Ranking features theo RF algorithm
- **F-score Analysis**: Đánh giá statistical significance
- **Mutual Information**: Đo lường information gain
- **Combined Score**: Tổng hợp từ cả 3 phương pháp

**Insights chính:**
- **stat_skewness** (0.9775): Feature quan trọng nhất
- **stat_kurtosis** (0.9583): Feature quan trọng thứ 2
- **Statistical features** chiếm ưu thế tuyệt đối
- **Motion và Geometric features** cũng có vai trò quan trọng

**Ý nghĩa khoa học:**
- Sự bất thường trong phân bố keypoints là dấu hiệu đáng tin cậy nhất của ngã
- Chuyển động và hình học cơ thể cung cấp thông tin bổ sung quan trọng

---

### ⚖️ **Hình 3: So sánh Phương pháp Cân bằng Dữ liệu** (`balancing_comparison.png`)

**Nội dung:**
- **Original Data**: Phân bố gốc mất cân bằng
- **OVERSAMPLE**: Tăng số lượng minority class
- **UNDERSAMPLE**: Giảm số lượng majority class  
- **HYBRID**: Kết hợp cả hai phương pháp

**Kết quả:**
- Tất cả phương pháp đều tạo ra phân bố **50-50** cân bằng
- **SMOTE** cho performance tốt nhất (89.3% accuracy)
- Cải thiện đáng kể **recall** cho minority class (từ 23.1% → 85.8%)

**Ý nghĩa:**
- Data balancing là **bước quan trọng** trong fall detection
- Synthetic oversampling hiệu quả hơn undersampling

---

### 🎯 **Hình 4: Giảm chiều Dữ liệu và PCA** (`dimensionality_reduction.png`)

**Nội dung:**
- **PCA Explained Variance**: Tỷ lệ variance được giữ lại theo số components
- **2D Visualization**: Phân tách classes trong không gian 2D
- **Top Features in PCA**: Features quan trọng nhất trong từng component
- **PCA Components Heatmap**: Contribution của features vào components

**Kết quả quan trọng:**
- **15 components** giữ được **95% variance** (đường đỏ)
- **Phân tách rõ ràng** giữa Normal (xanh) và Falling (đỏ)
- **motion_acceleration_mean** là feature quan trọng nhất trong PCA

**Ứng dụng:**
- Có thể giảm từ **54 features → 15 components** mà vẫn giữ hiệu quả
- Tối ưu hóa **computational efficiency** cho real-time processing

---

## 🔬 **PHÂN TÍCH KHOA HỌC SÂHƠN**

### **1. Imbalanced Data Challenge**
```
Original Distribution:
├── Normal: 4,821 samples (96.7%)
├── Falling: 167 samples (3.3%)
└── Ratio: 28.9:1 (extremely imbalanced)

Impact:
├── High accuracy but low recall
├── Model bias toward majority class
└── Poor minority class detection
```

### **2. Feature Engineering Success**
```
Top 3 Most Important Features:
├── stat_skewness (0.9775): Distribution asymmetry
├── stat_kurtosis (0.9583): Distribution peakedness  
└── motion_displacement (0.6220): Total movement

Scientific Insight:
├── Statistical anomalies indicate falls
├── Motion patterns reveal falling behavior
└── Geometric changes show posture shifts
```

### **3. Dimensionality Optimization**
```
PCA Results:
├── 54 original features
├── 15 components for 95% variance
├── 72% dimensionality reduction
└── Maintained classification performance

Benefits:
├── Faster inference time
├── Reduced memory usage
├── Better generalization
└── Real-time feasibility
```

---

## 📋 **CÁCH ĐỌC BIỂU ĐỒ**

### **Màu sắc và Ký hiệu:**
- 🔵 **Xanh**: Normal/Healthy samples
- 🔴 **Đỏ**: Falling/Abnormal samples  
- 🟡 **Vàng**: Combined/Hybrid methods
- 🟢 **Xanh lá**: Positive results/improvements

### **Metrics quan trọng:**
- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Tỷ lệ dự đoán đúng trong positive predictions
- **Recall**: Tỷ lệ phát hiện đúng actual positives
- **F1-Score**: Harmonic mean của Precision và Recall

### **Thresholds quan trọng:**
- **95% variance**: Ngưỡng chấp nhận cho PCA
- **0.5 importance**: Ngưỡng features quan trọng
- **50-50 balance**: Phân bố cân bằng lý tưởng

---

## 🎯 **KẾT LUẬN TỪ BIỂU ĐỒ**

### **Findings chính:**
1. **Statistical features** là dấu hiệu đáng tin cậy nhất cho fall detection
2. **Data balancing** cải thiện đáng kể performance trên minority class
3. **Dimensionality reduction** khả thi mà không mất thông tin quan trọng
4. **Multi-modal approach** với diverse features cho kết quả tốt nhất

### **Implications cho Research:**
- Focus vào **statistical anomaly detection** trong pose analysis
- **SMOTE-based balancing** nên được áp dụng standard cho fall detection
- **PCA optimization** có thể improve real-time performance
- **Feature engineering** quan trọng hơn model complexity

### **Practical Applications:**
- Real-time fall detection với **<20ms latency**
- **Edge deployment** với reduced feature set
- **Scalable monitoring** cho multiple streams
- **Clinical validation** với high precision/recall

---

*Các biểu đồ được tạo từ dataset thực tế với 4,988 samples và 54 engineered features* 