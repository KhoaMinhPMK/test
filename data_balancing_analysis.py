#%% Data Balancing & Feature Analysis cho Fall Detection
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("🚀 DATA BALANCING & FEATURE ANALYSIS")
print("=" * 50)
print("🎯 Mục tiêu:")
print("   ✅ Phân tích phân bố dữ liệu")
print("   ✅ Cân bằng dữ liệu (SMOTE, ADASYN, etc.)")
print("   ✅ Feature selection & importance")
print("   ✅ Dimensionality reduction")
print("   ✅ Visualization & insights")

#%% Cấu hình
CONFIG = {
    'input_file': 'extracted_features/extracted_features.csv',
    'output_dir': 'balanced_data',
    'analysis_dir': 'feature_analysis',
    'random_state': 42,
    'test_size': 0.2,
    'n_components_pca': 50,  # Số components cho PCA
    'top_k_features': 30,    # Top K features quan trọng nhất
}

# Tạo directories
for dir_name in [CONFIG['output_dir'], CONFIG['analysis_dir']]:
    os.makedirs(dir_name, exist_ok=True)

#%% Load và phân tích dữ liệu
def load_and_analyze_data():
    """Load dữ liệu và phân tích cơ bản"""
    
    print(f"\n📊 Loading data từ: {CONFIG['input_file']}")
    
    if not os.path.exists(CONFIG['input_file']):
        print(f"❌ File không tồn tại: {CONFIG['input_file']}")
        return None
    
    df = pd.read_csv(CONFIG['input_file'])
    
    print(f"📋 Dataset shape: {df.shape}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Phân tích phân bố labels
    label_counts = df['label'].value_counts()
    print(f"\n📊 Label distribution:")
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        label_name = "Normal" if label == 0 else "Falling"
        print(f"   {label_name} (label={label}): {count} samples ({percentage:.1f}%)")
    
    # Tính imbalance ratio
    imbalance_ratio = label_counts[0] / label_counts[1] if 1 in label_counts else float('inf')
    print(f"📊 Imbalance ratio (Normal:Falling): {imbalance_ratio:.1f}:1")
    
    # Phân tích missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n⚠️ Missing values detected:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"   {col}: {missing} ({missing/len(df)*100:.1f}%)")
    else:
        print(f"\n✅ Không có missing values")
    
    # Phân tích feature types
    feature_columns = [col for col in df.columns if col not in ['file_name', 'window_start', 'window_end', 'label', 'fall_ratio']]
    
    print(f"\n📋 Feature analysis:")
    print(f"   Total features: {len(feature_columns)}")
    
    feature_types = {}
    for col in feature_columns:
        prefix = col.split('_')[0]
        if prefix not in feature_types:
            feature_types[prefix] = []
        feature_types[prefix].append(col)
    
    for feat_type, features in feature_types.items():
        print(f"   {feat_type}: {len(features)} features")
    
    return df, feature_columns, feature_types

#%% Visualization functions
def plot_data_distribution(df):
    """Vẽ biểu đồ phân bố dữ liệu"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Label distribution
    label_counts = df['label'].value_counts()
    labels = ['Normal', 'Falling']
    colors = ['lightblue', 'lightcoral']
    
    axes[0, 0].pie(label_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Label Distribution', fontweight='bold')
    
    # 2. Fall ratio distribution
    axes[0, 1].hist(df['fall_ratio'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Fall Ratio Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Fall Ratio')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Window duration distribution
    if 'temporal_duration' in df.columns:
        axes[0, 2].hist(df['temporal_duration'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Window Duration Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Duration')
        axes[0, 2].set_ylabel('Frequency')
    
    # 4. Feature correlation heatmap (sample)
    feature_cols = [col for col in df.columns if col.startswith(('geo_', 'motion_', 'pose_'))][:10]
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation (Sample)', fontweight='bold')
    
    # 5. Box plot cho một số features quan trọng
    important_features = ['geo_aspect_ratio', 'motion_velocity_mean', 'pose_orientation']
    available_features = [f for f in important_features if f in df.columns]
    
    if available_features:
        df_melted = df[available_features + ['label']].melt(id_vars=['label'], var_name='feature', value_name='value')
        sns.boxplot(data=df_melted, x='feature', y='value', hue='label', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Distribution by Label', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Dataset size by file
    if 'file_name' in df.columns:
        file_counts = df['file_name'].value_counts().head(10)
        axes[1, 2].bar(range(len(file_counts)), file_counts.values, color='orange', alpha=0.7)
        axes[1, 2].set_title('Top 10 Files by Sample Count', fontweight='bold')
        axes[1, 2].set_xlabel('File Index')
        axes[1, 2].set_ylabel('Sample Count')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['analysis_dir']}/data_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

#%% Feature importance analysis
def analyze_feature_importance(X, y, feature_names):
    """Phân tích tầm quan trọng của features"""
    
    print(f"\n🔍 Analyzing feature importance...")
    
    # 1. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # 2. Univariate Feature Selection (F-score)
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X, y)
    f_scores = f_selector.scores_
    
    # 3. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=CONFIG['random_state'])
    
    # Tạo DataFrame kết quả
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importance,
        'f_score': f_scores,
        'mutual_info': mi_scores
    })
    
    # Normalize scores
    for col in ['rf_importance', 'f_score', 'mutual_info']:
        importance_df[f'{col}_norm'] = (importance_df[col] - importance_df[col].min()) / (importance_df[col].max() - importance_df[col].min())
    
    # Combined score
    importance_df['combined_score'] = (
        importance_df['rf_importance_norm'] + 
        importance_df['f_score_norm'] + 
        importance_df['mutual_info_norm']
    ) / 3
    
    # Sort by combined score
    importance_df = importance_df.sort_values('combined_score', ascending=False)
    
    # Lưu kết quả
    importance_df.to_csv(f"{CONFIG['analysis_dir']}/feature_importance.csv", index=False)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top features by different methods
    top_n = min(20, len(importance_df))
    
    # Random Forest
    top_rf = importance_df.head(top_n)
    axes[0, 0].barh(range(top_n), top_rf['rf_importance'], color='skyblue')
    axes[0, 0].set_yticks(range(top_n))
    axes[0, 0].set_yticklabels(top_rf['feature'], fontsize=8)
    axes[0, 0].set_title('Top Features - Random Forest Importance', fontweight='bold')
    axes[0, 0].invert_yaxis()
    
    # F-score
    top_f = importance_df.nlargest(top_n, 'f_score')
    axes[0, 1].barh(range(top_n), top_f['f_score'], color='lightgreen')
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(top_f['feature'], fontsize=8)
    axes[0, 1].set_title('Top Features - F-score', fontweight='bold')
    axes[0, 1].invert_yaxis()
    
    # Mutual Information
    top_mi = importance_df.nlargest(top_n, 'mutual_info')
    axes[1, 0].barh(range(top_n), top_mi['mutual_info'], color='lightcoral')
    axes[1, 0].set_yticks(range(top_n))
    axes[1, 0].set_yticklabels(top_mi['feature'], fontsize=8)
    axes[1, 0].set_title('Top Features - Mutual Information', fontweight='bold')
    axes[1, 0].invert_yaxis()
    
    # Combined Score
    top_combined = importance_df.head(top_n)
    axes[1, 1].barh(range(top_n), top_combined['combined_score'], color='gold')
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels(top_combined['feature'], fontsize=8)
    axes[1, 1].set_title('Top Features - Combined Score', fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['analysis_dir']}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # In top features
    print(f"\n🏆 Top {CONFIG['top_k_features']} most important features:")
    for i, row in importance_df.head(CONFIG['top_k_features']).iterrows():
        print(f"   {row['feature']}: {row['combined_score']:.4f}")
    
    return importance_df

#%% Data balancing techniques
def balance_data(X, y, method='smote'):
    """Cân bằng dữ liệu bằng các phương pháp khác nhau"""
    
    print(f"\n⚖️ Balancing data using {method.upper()}...")
    
    original_distribution = Counter(y)
    print(f"📊 Original distribution: {original_distribution}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=CONFIG['random_state'])
    elif method == 'adasyn':
        sampler = ADASYN(random_state=CONFIG['random_state'])
    elif method == 'borderline_smote':
        sampler = BorderlineSMOTE(random_state=CONFIG['random_state'])
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=CONFIG['random_state'])
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=CONFIG['random_state'])
    elif method == 'random_undersample':
        sampler = RandomUnderSampler(random_state=CONFIG['random_state'])
    else:
        print(f"❌ Unknown method: {method}")
        return X, y
    
    try:
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        balanced_distribution = Counter(y_balanced)
        
        print(f"📊 Balanced distribution: {balanced_distribution}")
        print(f"📊 Size change: {len(X)} → {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
        
    except Exception as e:
        print(f"❌ Error in {method}: {e}")
        return X, y

#%% Dimensionality reduction
def reduce_dimensions(X, y, feature_names):
    """Giảm chiều dữ liệu và visualization"""
    
    print(f"\n📉 Reducing dimensions...")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=CONFIG['n_components_pca'], random_state=CONFIG['random_state'])
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"📊 PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # t-SNE (chỉ với subset nhỏ để tránh quá lâu)
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X_tsne_input = X_scaled[indices]
        y_tsne = y[indices]
    else:
        X_tsne_input = X_scaled
        y_tsne = y
    
    tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=30)
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[0, 0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    axes[0, 0].set_title('PCA Explained Variance', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Cumulative Explained Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCA 2D
    axes[0, 1].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
    axes[0, 1].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', alpha=0.6, label='Falling', s=20)
    axes[0, 1].set_title('PCA 2D Visualization', fontweight='bold')
    axes[0, 1].set_xlabel('First Principal Component')
    axes[0, 1].set_ylabel('Second Principal Component')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # t-SNE 2D
    axes[1, 0].scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1], c='blue', alpha=0.6, label='Normal', s=20)
    axes[1, 0].scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1], c='red', alpha=0.6, label='Falling', s=20)
    axes[1, 0].set_title('t-SNE 2D Visualization', fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].legend()
    
    # Feature importance in PCA
    feature_importance_pca = np.abs(pca.components_).mean(axis=0)
    top_features_idx = np.argsort(feature_importance_pca)[-20:]
    
    axes[1, 1].barh(range(20), feature_importance_pca[top_features_idx], color='purple', alpha=0.7)
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([feature_names[i] for i in top_features_idx], fontsize=8)
    axes[1, 1].set_title('Top Features in PCA', fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['analysis_dir']}/dimensionality_reduction.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, pca, scaler

#%% Main processing function
def process_data_balancing_analysis():
    """Main function để xử lý cân bằng dữ liệu và phân tích"""
    
    # 1. Load và phân tích dữ liệu
    result = load_and_analyze_data()
    if result is None:
        return
    
    df, feature_columns, feature_types = result
    
    # 2. Visualization
    plot_data_distribution(df)
    
    # 3. Prepare data
    X = df[feature_columns].fillna(0)  # Fill NaN với 0
    y = df['label'].values
    
    # 4. Feature importance analysis
    importance_df = analyze_feature_importance(X, y, feature_columns)
    
    # 5. Select top features
    top_features = importance_df.head(CONFIG['top_k_features'])['feature'].tolist()
    X_selected = X[top_features]
    
    print(f"\n🎯 Selected {len(top_features)} top features")
    
    # 6. Dimensionality reduction
    X_pca, pca, scaler = reduce_dimensions(X_selected, y, top_features)
    
    # 7. Data balancing với nhiều methods
    balancing_methods = ['smote', 'adasyn', 'borderline_smote', 'smote_tomek']
    
    balanced_datasets = {}
    
    for method in balancing_methods:
        print(f"\n{'='*50}")
        X_balanced, y_balanced = balance_data(X_selected, y, method)
        
        # Lưu balanced dataset
        balanced_df = pd.DataFrame(X_balanced, columns=top_features)
        balanced_df['label'] = y_balanced
        
        # Thêm metadata
        balanced_df['balancing_method'] = method
        
        output_file = f"{CONFIG['output_dir']}/balanced_data_{method}.csv"
        balanced_df.to_csv(output_file, index=False)
        
        balanced_datasets[method] = (X_balanced, y_balanced)
        
        print(f"💾 Saved: {output_file}")
    
    # 8. Comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original data
    axes[0].pie(Counter(y).values(), labels=['Normal', 'Falling'], autopct='%1.1f%%', 
                colors=['lightblue', 'lightcoral'], startangle=90)
    axes[0].set_title('Original Data', fontweight='bold')
    
    # Balanced datasets
    for i, (method, (X_bal, y_bal)) in enumerate(balanced_datasets.items(), 1):
        if i < len(axes):
            axes[i].pie(Counter(y_bal).values(), labels=['Normal', 'Falling'], autopct='%1.1f%%',
                       colors=['lightblue', 'lightcoral'], startangle=90)
            axes[i].set_title(f'{method.upper()}', fontweight='bold')
    
    # Hide unused subplot
    if len(balanced_datasets) + 1 < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['analysis_dir']}/balancing_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Summary report
    print(f"\n📋 SUMMARY REPORT")
    print(f"=" * 50)
    print(f"📊 Original dataset: {len(df)} samples")
    print(f"📊 Features: {len(feature_columns)} → {len(top_features)} (selected)")
    print(f"📊 Imbalance ratio: {Counter(y)[0]/Counter(y)[1]:.1f}:1")
    print(f"📁 Output directory: {CONFIG['output_dir']}")
    print(f"📁 Analysis directory: {CONFIG['analysis_dir']}")
    
    print(f"\n🎯 Top 10 most important features:")
    for i, feature in enumerate(top_features[:10], 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n⚖️ Balanced datasets created:")
    for method in balancing_methods:
        print(f"   ✅ {method}: balanced_data_{method}.csv")
    
    return df, balanced_datasets, importance_df, top_features

if __name__ == "__main__":
    # Chạy phân tích và cân bằng dữ liệu
    result = process_data_balancing_analysis()
    
    if result:
        df, balanced_datasets, importance_df, top_features = result
        print(f"\n🎉 Data balancing & analysis hoàn thành!")
    else:
        print(f"\n❌ Có lỗi xảy ra trong quá trình xử lý!") 