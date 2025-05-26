#%% Import thư viện cần thiết
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import random
from tqdm import tqdm
import multiprocessing
from datetime import datetime

# Cấu hình GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU được phát hiện và cấu hình: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")

# Tắt Mixed Precision để tiết kiệm memory
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
print(f"🎯 Mixed Precision disabled để tiết kiệm memory")

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"CPU cores: {multiprocessing.cpu_count()}")

#%% Cấu hình tham số cho Person Detection + Pose Model
CONFIG = {
    # Đường dẫn dữ liệu - CẬP NHẬT THEO ĐƯỜNG DẪN MỚI
    'base_path': '/workspace/khoaminh/data',
    'datasets': ['Coffee_room_01', 'Coffee_room_02', 'Home_01', 'Home_02'],
    
    # Tham số hình ảnh
    'input_size': (416, 416),      # Kích thước đầu vào YOLO-style
    'num_keypoints': 17,           # COCO pose: 17 keypoints
    'grid_size': (13, 13),         # YOLO grid size
    'num_anchors': 3,              # Số anchor boxes
    'sequence_length': 8,          # Số frames liên tiếp cho LSTM
    'temporal_stride': 2,          # Bước nhảy giữa các frames
    
    # YOLO-style anchors (scaled for 416x416)
    'anchors': [
        [10, 13], [16, 30], [33, 23],    # Small objects
        [30, 61], [62, 45], [59, 119],   # Medium objects  
        [116, 90], [156, 198], [373, 326] # Large objects
    ],
    
    # Tham số training
    'batch_size': 4,               # Giảm thêm batch size để tiết kiệm memory
    'epochs': 50,                  # Giảm epochs
    'learning_rate': 5e-5,         # Giảm learning rate
    'validation_split': 0.2,
    
    # Tham số loss weights - Cân bằng lại
    'lambda_bbox': 5.0,            # Giảm weight cho bbox
    'lambda_pose': 3.0,            # Giảm weight cho pose
    'lambda_conf': 1.0,            # Giữ nguyên confidence
    
    # Regularization
    'l1_reg': 1e-5,                # L1 regularization
    'l2_reg': 1e-4,                # L2 regularization
    
    # Tham số khác
    'max_samples_per_video': 200,  # Giảm samples để tránh overfit
    'confidence_threshold': 0.5,
    'save_model_path': 'models/',
    
    # Data augmentation
    'use_augmentation': True,      # Bật augmentation
}

# COCO Pose keypoints
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections cho vẽ pose
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Chân
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # Thân + tay
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],         # Tay + mắt
    [2, 4], [3, 5], [4, 6], [5, 7]                    # Mặt + vai
]

print(f"📋 Person Detection + Pose Configuration:")
for key, value in CONFIG.items():
        print(f"   {key}: {value}")

#%% Hàm tìm và load dữ liệu
def find_data_files():
    """Tìm tất cả file video và annotation"""
    base_path = CONFIG['base_path']
    datasets = CONFIG['datasets']
    
    print(f"🔍 Tìm kiếm dữ liệu trong: {base_path}")
    
    matched_data = []
    
    for dataset_name in datasets:
        print(f"\n📁 Xử lý dataset: {dataset_name}")
        
        # Đường dẫn video và annotation - CẬP NHẬT THEO CẤU TRÚC MỚI
        video_dir = f"{base_path}/{dataset_name}/{dataset_name}/Videos"
        annotation_dir = f"{base_path}/{dataset_name}/{dataset_name}/Annotation_files_processed"
        
        # Tìm files
        video_patterns = [
            f"{video_dir}/*.avi",
            f"{video_dir}/*.mp4",
            f"{video_dir}/*.mov",
        ]
        
        video_files = []
        for pattern in video_patterns:
            video_files.extend(glob.glob(pattern))
        
        annotation_files = glob.glob(f"{annotation_dir}/*_with_pose.txt")
        
        print(f"   🎬 Videos: {len(video_files)}")
        print(f"   📝 Annotations: {len(annotation_files)}")
        
        # Ghép video với annotation
        for ann_file in annotation_files:
            ann_basename = os.path.basename(ann_file)
            video_name_from_ann = ann_basename.replace('_with_pose.txt', '')
            video_name_expected = video_name_from_ann.replace('_', ' ')
            
            possible_names = [
                f"{video_name_expected}.avi",
                f"{video_name_expected}.mp4",
                f"{video_name_from_ann}.avi",
                f"{video_name_from_ann}.mp4",
            ]
            
            matching_videos = []
            for video_file in video_files:
                video_basename = os.path.basename(video_file)
                if video_basename in possible_names:
                    matching_videos.append(video_file)
            
            if matching_videos:
                matched_data.append({
                    'dataset': dataset_name,
                    'video_name': video_name_from_ann,
                    'video_path': matching_videos[0],
                    'annotation_path': ann_file
                })
    
    print(f"\n📊 Tổng kết: {len(matched_data)} cặp video-annotation được tìm thấy")
    return matched_data

#%% Hàm parse annotation
def parse_annotation_file(annotation_path):
    """Parse file annotation để lấy bbox và pose keypoints"""
    annotations = {}
    
    try:
        with open(annotation_path, 'r') as f:
            fall_start = int(f.readline().strip())
            fall_end = int(f.readline().strip())
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_num = int(parts[0])
                    label = int(parts[1])
                    
                    # Bbox coordinates
                    bbox = {
                        'x1': int(parts[2]),
                        'y1': int(parts[3]),
                        'x2': int(parts[4]),
                        'y2': int(parts[5])
                    }
                    
                    # Parse pose keypoints
                    keypoints = []
                    for i in range(6, len(parts)):
                        if ':' in parts[i]:
                            kp_parts = parts[i].split(':')
                            if len(kp_parts) == 4:
                                x = float(kp_parts[1])
                                y = float(kp_parts[2])
                                conf = float(kp_parts[3])
                                keypoints.extend([x, y, conf])
                    
                    # Đảm bảo có đủ 17 keypoints (51 values)
                    while len(keypoints) < 51:
                        keypoints.extend([0, 0, 0])
                    keypoints = keypoints[:51]
                    
                    annotations[frame_num] = {
                        'bbox': bbox,
                        'keypoints': keypoints,
                        'label': label
                    }
        
        return annotations, fall_start, fall_end
        
    except Exception as e:
        print(f"❌ Lỗi parse {annotation_path}: {e}")
        return {}, -1, -1

#%% Hàm chuẩn hóa dữ liệu
def normalize_bbox(bbox, img_width, img_height):
    """Chuẩn hóa bbox về [0,1]"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Normalize to [0,1]
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]

def normalize_keypoints(keypoints, img_width, img_height):
    """Normalize keypoints coordinates"""
    normalized_kp = []
    for i in range(0, len(keypoints), 3):
        if i+2 < len(keypoints):
            x = keypoints[i] / img_width
            y = keypoints[i+1] / img_height
            conf = keypoints[i+2]
            normalized_kp.extend([x, y, conf])
    
    return normalized_kp

#%% Data Augmentation
@tf.function
def augment_image_and_targets(image, bbox, keypoints):
    """Augmentation cho ảnh và targets"""
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        # Flip bbox
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox = tf.stack([1.0 - x2, y1, 1.0 - x1, y2])
        
        # Flip keypoints (swap left-right pairs)
        kp_flipped = tf.identity(keypoints)
        # Flip x coordinates
        for i in range(0, 51, 3):
            kp_flipped = tf.tensor_scatter_nd_update(
                kp_flipped, [[i]], [1.0 - keypoints[i]]
            )
        keypoints = kp_flipped
    
    return image, bbox, keypoints

#%% Tạo dataset
def create_person_pose_dataset(data_files):
    """Tạo dataset cho Person Detection + Pose training"""
    
    def data_generator():
        print("🔄 Tạo Person Detection + Pose dataset...")
        sample_count = 0
        
        for data_item in tqdm(data_files, desc="Processing videos"):
            video_path = data_item['video_path']
            annotation_path = data_item['annotation_path']
            
            # Parse annotations
            annotations, fall_start, fall_end = parse_annotation_file(annotation_path)
            if not annotations:
                continue
            
            # Mở video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Sample frames có annotation
            annotated_frames = list(annotations.keys())
            max_samples = min(len(annotated_frames), CONFIG['max_samples_per_video'])
            sampled_frames = random.sample(annotated_frames, max_samples)
            
            for frame_num in sampled_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Resize frame
                frame_resized = cv2.resize(frame, CONFIG['input_size'])
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                
                # Get annotation
                ann = annotations[frame_num]
                
                # Normalize bbox và keypoints
                bbox_norm = normalize_bbox(ann['bbox'], frame_width, frame_height)
                keypoints_norm = normalize_keypoints(ann['keypoints'], frame_width, frame_height)
                
                # Confidence (có người = 1.0)
                confidence = 1.0
                
                # Data augmentation nếu được bật
                if CONFIG.get('use_augmentation', False) and random.random() > 0.5:
                    # Áp dụng augmentation đơn giản
                    frame_normalized = np.clip(frame_normalized + np.random.normal(0, 0.02, frame_normalized.shape), 0, 1)
                
                yield frame_normalized, bbox_norm, keypoints_norm, confidence
                sample_count += 1
            
            cap.release()
        
        print(f"✅ Dataset hoàn thành: {sample_count} samples")
    
    # Tạo tf.data.Dataset với format phù hợp cho multi-output model
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(*CONFIG['input_size'], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),  # bbox
            tf.TensorSpec(shape=(51,), dtype=tf.float32), # keypoints
            tf.TensorSpec(shape=(), dtype=tf.float32)     # confidence
        )
    )
    
    # Chuyển đổi format cho multi-output model: (x, (y1, y2, y3))
    def reformat_data(image, bbox, keypoints, confidence):
        return image, {
            'bbox_output': bbox,
            'pose_output': keypoints, 
            'conf_output': confidence
        }
    
    dataset = dataset.map(reformat_data)
    
    return dataset

#%% YOLO-Inspired Intelligent Model với LSTM và Advanced Features
def create_yolo_inspired_intelligent_model():
    """Tạo model kết hợp YOLO architecture với intelligent reasoning"""
    
    # Regularizers
    l1_reg = keras.regularizers.L1(CONFIG['l1_reg'])
    l2_reg = keras.regularizers.L2(CONFIG['l2_reg'])
    l1_l2_reg = keras.regularizers.L1L2(l1=CONFIG['l1_reg'], l2=CONFIG['l2_reg'])
    
    inputs = keras.Input(shape=(*CONFIG['input_size'], 3))
    
    # ===== DARKNET-INSPIRED BACKBONE =====
    def conv_bn_leaky(x, filters, kernel_size, strides=1, alpha=0.1):
        """Conv + BatchNorm + LeakyReLU block (YOLO style)"""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                         kernel_regularizer=l2_reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=alpha)(x)
        return x
    
    def residual_block(x, filters):
        """Residual block với YOLO style"""
        shortcut = x
        x = conv_bn_leaky(x, filters//2, 1)
        x = conv_bn_leaky(x, filters, 3)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                                   kernel_regularizer=l2_reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        return x
    
    # Stage 1: Initial convolution (giảm filters)
    x = conv_bn_leaky(inputs, 16, 3)  # Giảm từ 32 xuống 16
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Stage 2: 32 filters (giảm từ 64)
    x = conv_bn_leaky(x, 32, 3)  # Giảm từ 64 xuống 32
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Stage 3: 64 filters với residual blocks (giảm filters và blocks)
    x = conv_bn_leaky(x, 64, 3)  # Giảm từ 128 xuống 64
    # Bỏ residual blocks để giảm complexity
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Stage 4: 128 filters (giảm mạnh)
    x = conv_bn_leaky(x, 128, 3)  # Giảm từ 256 xuống 128
    x = residual_block(x, 128)  # Chỉ 1 block
    route_128 = x  # Save for skip connection
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Stage 5: 256 filters (giảm mạnh)
    x = conv_bn_leaky(x, 256, 3)  # Giảm từ 512 xuống 256
    x = residual_block(x, 256)  # Chỉ 1 block
    route_256 = x  # Save for skip connection
    x = layers.MaxPooling2D(2, strides=2)(x)
    
    # Stage 6: 256 filters (giảm mạnh từ 512)
    x = conv_bn_leaky(x, 256, 3)  # Giảm từ 512 xuống 256
    # Bỏ residual blocks
    
    # ===== YOLO-STYLE DETECTION HEAD (giảm mạnh kích thước) =====
    # Detection layers với filters rất nhỏ
    x = conv_bn_leaky(x, 128, 1)  # Giảm từ 256 xuống 128
    x = conv_bn_leaky(x, 256, 3)  # Giảm từ 512 xuống 256
    x = conv_bn_leaky(x, 128, 1)  # Giảm từ 256 xuống 128
    
    # ===== SPATIAL ATTENTION =====
    attention = layers.Conv2D(1, 1, activation='sigmoid', name='spatial_attention')(x)
    x = layers.Multiply()([x, attention])
    
    # ===== MULTI-SCALE FEATURE FUSION (giảm mạnh kích thước) =====
    # Current scale features
    current_features = layers.GlobalAveragePooling2D()(x)
    
    # Medium scale features (from route_256) - giảm kích thước
    medium_features = layers.GlobalAveragePooling2D()(route_256)
    medium_features = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(medium_features)  # Giảm từ 128 xuống 64
    
    # Large scale features (from route_128) - giảm kích thước
    large_features = layers.GlobalAveragePooling2D()(route_128)
    large_features = layers.Dense(32, activation='relu', kernel_regularizer=l2_reg)(large_features)  # Giảm từ 64 xuống 32
    
    # Fuse multi-scale features
    fused_features = layers.Concatenate()([current_features, medium_features, large_features])
    
    # ===== TEMPORAL REASONING với LSTM =====
    # Reshape cho LSTM temporal reasoning
    feature_dim = fused_features.shape[-1]
    temporal_chunks = 8
    chunk_size = feature_dim // temporal_chunks
    
    # Pad nếu cần thiết
    if feature_dim % temporal_chunks != 0:
        padding_size = temporal_chunks - (feature_dim % temporal_chunks)
        padding = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, padding_size]]))(fused_features)
        feature_dim = padding.shape[-1]
        chunk_size = feature_dim // temporal_chunks
        temporal_features = layers.Reshape((temporal_chunks, chunk_size))(padding)
    else:
        temporal_features = layers.Reshape((temporal_chunks, chunk_size))(fused_features)
    
    # Bidirectional LSTM layers (giảm mạnh kích thước)
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                   kernel_regularizer=l1_l2_reg, recurrent_regularizer=l2_reg)  # Giảm từ 128 xuống 64
    )(temporal_features)
    
    lstm_out = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
                   kernel_regularizer=l1_l2_reg, recurrent_regularizer=l2_reg)  # Giảm từ 64 xuống 32
    )(lstm_out)
    
    # ===== INTELLIGENT FUSION =====
    combined_features = layers.Concatenate()([fused_features, lstm_out])
    
    # Deep reasoning với progressive dropout (giảm mạnh kích thước)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l1_l2_reg)(combined_features)  # Giảm từ 1024 xuống 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=l1_l2_reg)(x)  # Giảm từ 512 xuống 128
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # ===== YOLO-STYLE MULTI-TASK OUTPUTS (giảm mạnh kích thước) =====
    # 1. Bbox regression (YOLO format: cx, cy, w, h)
    bbox_branch = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg, name='bbox_reasoning')(x)  # Giảm từ 128 xuống 64
    bbox_branch = layers.Dropout(0.3)(bbox_branch)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output', dtype='float32')(bbox_branch)
    
    # 2. Pose keypoints với spatial reasoning
    pose_branch = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg, name='pose_reasoning')(x)  # Giảm từ 256 xuống 128
    pose_branch = layers.Dropout(0.3)(pose_branch)
    pose_output = layers.Dense(51, activation='sigmoid', name='pose_output', dtype='float32')(pose_branch)
    
    # 3. Objectness confidence (YOLO style)
    conf_branch = layers.Dense(32, activation='relu', kernel_regularizer=l2_reg, name='conf_reasoning')(x)  # Giảm từ 64 xuống 32
    conf_branch = layers.Dropout(0.2)(conf_branch)
    conf_output = layers.Dense(1, activation='sigmoid', name='conf_output', dtype='float32')(conf_branch)
    
    model = keras.Model(
        inputs=inputs, 
        outputs=[bbox_output, pose_output, conf_output],
        name='YOLOInspiredUltraLightDetector'  # Đổi tên để phản ánh việc tối ưu cực mạnh
    )
    
    return model

#%% Custom Loss Functions
def bbox_loss(y_true, y_pred):
    """Smooth L1 loss cho bbox regression"""
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    smooth_l1_loss = (less_than_one * 0.5 * diff**2) + (1.0 - less_than_one) * (diff - 0.5)
    return tf.reduce_mean(smooth_l1_loss)

def pose_loss(y_true, y_pred):
    """MSE loss cho pose keypoints với confidence weighting"""
    # Extract confidence values (every 3rd element starting from index 2)
    confidence_mask = y_true[..., 2::3]  # [batch, 17]
    
    # Reshape để tính loss
    y_true_reshaped = tf.reshape(y_true, [-1, 17, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 17, 3])
    
    # Chỉ tính loss cho keypoints có confidence > 0
    valid_mask = tf.cast(confidence_mask > 0, tf.float32)
    valid_mask = tf.expand_dims(valid_mask, -1)  # [batch, 17, 1]
    
    # MSE loss
    mse = tf.square(y_true_reshaped - y_pred_reshaped)
    weighted_mse = mse * valid_mask
    
    return tf.reduce_mean(weighted_mse)

def confidence_loss(y_true, y_pred):
    """Binary crossentropy cho confidence"""
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

#%% Training function
def train_person_pose_model():
    """Training Person Detection + Pose model"""
    
    print("🚀 Bắt đầu training Person Detection + Pose model...")
    
    # Tìm dữ liệu
    data_files = find_data_files()
    if not data_files:
        print("❌ Không tìm thấy dữ liệu!")
        return None, None
    
    # Chia train/validation
    train_files, val_files = train_test_split(
        data_files, 
        test_size=CONFIG['validation_split'], 
        random_state=42
    )
    
    print(f"📊 Chia dữ liệu: {len(train_files)} train, {len(val_files)} validation")
    
    # Tạo datasets
    print("📦 Tạo datasets...")
    train_dataset_raw = create_person_pose_dataset(train_files)
    val_dataset_raw = create_person_pose_dataset(val_files)
    
    # Tối ưu dataset
    train_dataset = train_dataset_raw.shuffle(1000).batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset_raw.batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    # Tạo model
    model = create_yolo_inspired_intelligent_model()
    
    # Compile model với custom losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss={
            'bbox_output': bbox_loss,
            'pose_output': pose_loss,
            'conf_output': confidence_loss
        },
        loss_weights={
            'bbox_output': CONFIG['lambda_bbox'],
            'pose_output': CONFIG['lambda_pose'],
            'conf_output': CONFIG['lambda_conf']
        },
        metrics={
            'bbox_output': 'mae',
            'pose_output': 'mae',
            'conf_output': 'accuracy'
        }
    )
    
    print("🏗️ YOLO-Inspired Intelligent Model architecture:")
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"{CONFIG['save_model_path']}YOLOInspiredUltraLight_{timestamp}.h5"
    
    os.makedirs(CONFIG['save_model_path'], exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,              # Giảm LR mạnh hơn
            patience=5,              # Giảm patience
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,             # Giảm patience để stop sớm hơn
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Training
    print("🔥 Bắt đầu training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"✅ Training hoàn thành! Model đã lưu tại: {model_save_path}")
    
    return model, history

#%% Inference và visualization
def draw_pose_skeleton(image, keypoints, confidence_threshold=0.3):
    """Vẽ skeleton pose lên ảnh"""
    
    img_height, img_width = image.shape[:2]
    
    # Vẽ keypoints
    for i in range(0, len(keypoints), 3):
        if i+2 < len(keypoints):
            x = int(keypoints[i] * img_width)
            y = int(keypoints[i+1] * img_height)
            conf = keypoints[i+2]
            
            if conf > confidence_threshold:
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
    
    # Vẽ skeleton
    for connection in SKELETON:
        kp1_idx = (connection[0] - 1) * 3  # COCO index bắt đầu từ 1
        kp2_idx = (connection[1] - 1) * 3
        
        if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
            x1 = int(keypoints[kp1_idx] * img_width)
            y1 = int(keypoints[kp1_idx + 1] * img_height)
            conf1 = keypoints[kp1_idx + 2]
            
            x2 = int(keypoints[kp2_idx] * img_width)
            y2 = int(keypoints[kp2_idx + 1] * img_height)
            conf2 = keypoints[kp2_idx + 2]
            
            if conf1 > confidence_threshold and conf2 > confidence_threshold:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return image

def predict_and_visualize(model, test_files, num_samples=3):
    """Predict và visualize kết quả"""
    
    print(f"🧪 Testing Person Detection + Pose với {num_samples} samples...")
    
    for data_item in test_files[:num_samples]:
        video_path = data_item['video_path']
        annotation_path = data_item['annotation_path']
        
        # Parse annotations
        annotations, _, _ = parse_annotation_file(annotation_path)
        
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        
        # Lấy một frame ngẫu nhiên có annotation
        annotated_frames = list(annotations.keys())
        if not annotated_frames:
            cap.release()
            continue
            
        random_frame = random.choice(annotated_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        
        if not ret:
            cap.release()
            continue
        
        # Preprocess
        frame_resized = cv2.resize(frame, CONFIG['input_size'])
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        input_data = np.expand_dims(frame_normalized, axis=0)
        
        # Predict
        bbox_pred, pose_pred, conf_pred = model.predict(input_data, verbose=0)
        
        # Extract predictions
        bbox = bbox_pred[0]
        pose = pose_pred[0]
        confidence = conf_pred[0][0]
        
        # Convert bbox to pixel coordinates
        img_h, img_w = CONFIG['input_size']
        x1 = int(bbox[0] * img_w)
        y1 = int(bbox[1] * img_h)
        x2 = int(bbox[2] * img_w)
        y2 = int(bbox[3] * img_h)
        
        # Visualize
        result_img = frame_resized.copy()
        
        # Vẽ predicted bbox
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(result_img, f'Conf: {confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Vẽ predicted pose
        result_img = draw_pose_skeleton(result_img, pose)
        
        # Ground truth
        gt_ann = annotations[random_frame]
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Scale ground truth bbox
        gt_bbox = gt_ann['bbox']
        gt_x1 = int(gt_bbox['x1'] * img_w / frame_width)
        gt_y1 = int(gt_bbox['y1'] * img_h / frame_height)
        gt_x2 = int(gt_bbox['x2'] * img_w / frame_width)
        gt_y2 = int(gt_bbox['y2'] * img_h / frame_height)
        
        gt_img = frame_resized.copy()
        cv2.rectangle(gt_img, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)
        
        # Scale ground truth keypoints
        gt_keypoints = normalize_keypoints(gt_ann['keypoints'], frame_width, frame_height)
        gt_img = draw_pose_skeleton(gt_img, gt_keypoints)
        
        # Display
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        plt.title('Original Frame')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Prediction (Conf: {confidence:.2f})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        cap.release()

#%% Visualize training history
def plot_training_history(history):
    """Vẽ biểu đồ training history"""
    
    plt.figure(figsize=(15, 10))
    
    # Total loss
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Total Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bbox loss
    plt.subplot(2, 3, 2)
    plt.plot(history.history['bbox_output_loss'], label='Train Bbox Loss', linewidth=2)
    plt.plot(history.history['val_bbox_output_loss'], label='Val Bbox Loss', linewidth=2)
    plt.title('Bbox Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Pose loss
    plt.subplot(2, 3, 3)
    plt.plot(history.history['pose_output_loss'], label='Train Pose Loss', linewidth=2)
    plt.plot(history.history['val_pose_output_loss'], label='Val Pose Loss', linewidth=2)
    plt.title('Pose Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence loss
    plt.subplot(2, 3, 4)
    plt.plot(history.history['conf_output_loss'], label='Train Conf Loss', linewidth=2)
    plt.plot(history.history['val_conf_output_loss'], label='Val Conf Loss', linewidth=2)
    plt.title('Confidence Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bbox MAE
    plt.subplot(2, 3, 5)
    plt.plot(history.history['bbox_output_mae'], label='Train Bbox MAE', linewidth=2)
    plt.plot(history.history['val_bbox_output_mae'], label='Val Bbox MAE', linewidth=2)
    plt.title('Bbox MAE', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence accuracy
    plt.subplot(2, 3, 6)
    plt.plot(history.history['conf_output_accuracy'], label='Train Conf Acc', linewidth=2)
    plt.plot(history.history['val_conf_output_accuracy'], label='Val Conf Acc', linewidth=2)
    plt.title('Confidence Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Person Detection + Pose Training Results', fontsize=16, fontweight='bold', y=1.02)
    plt.show()

#%% Main execution
if __name__ == "__main__":
    print("🎯 YOLO-INSPIRED ULTRA-LIGHT DETECTION + POSE ESTIMATION")
    print("=" * 65)
    print("🧠 Model kết hợp YOLO architecture với intelligent reasoning (ultra-light)!")
    print("📋 Features:")
    print("   ✅ DarkNet-inspired backbone (YOLO style)")
    print("   ✅ Conv + BatchNorm + LeakyReLU blocks")
    print("   ✅ Multi-scale skip connections")
    print("   ✅ YOLO-style detection head")
    print("   ✅ Bidirectional LSTM temporal reasoning")
    print("   ✅ Spatial attention mechanism")
    print("   ✅ L1 + L2 regularization")
    print("   ✅ Multi-scale feature fusion")
    print("   ✅ Progressive dropout strategies")
    print("   ✅ YOLO-inspired multi-task learning")
    print("   🔧 Ultra-light architecture (batch_size=4)")
    print("   🔧 Mixed Precision disabled để tiết kiệm memory")
    
    # Training
    model, history = train_person_pose_model()
    
    if model is not None and history is not None:
        # Vẽ biểu đồ training
        plot_training_history(history)
        
        # Test model
        data_files = find_data_files()
        if data_files:
            predict_and_visualize(model, data_files, num_samples=3)
    
    print("\n🎉 YOLO-Inspired Ultra-Light Detection + Pose training hoàn thành!") 