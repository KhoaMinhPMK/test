#%% Feature Extraction Algorithm cho Fall Detection
import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 FEATURE EXTRACTION ALGORITHM")
print("=" * 50)
print("🎯 Mục tiêu: Trích xuất đặc trưng toàn diện cho Fall Detection")
print("📋 Loại đặc trưng:")
print("   ✅ Geometric Features (Hình học)")
print("   ✅ Motion Features (Chuyển động)")
print("   ✅ Pose Features (Tư thế)")
print("   ✅ Temporal Features (Thời gian)")
print("   ✅ Statistical Features (Thống kê)")
print("   ✅ Frequency Features (Tần số)")

#%% Cấu hình
CONFIG = {
    'input_dir': 'new_dataset',
    'output_dir': 'extracted_features',
    'window_size': 10,  # Cửa sổ thời gian để tính đặc trưng temporal
    'overlap': 5,       # Overlap giữa các windows
    'min_confidence': 0.3,  # Confidence threshold cho keypoints
}

# COCO Pose keypoints mapping
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Nhóm keypoints theo chức năng
KEYPOINT_GROUPS = {
    'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
    'torso': [5, 6, 11, 12],  # shoulders, hips
    'arms': [5, 6, 7, 8, 9, 10],  # shoulders, elbows, wrists
    'legs': [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
}

print(f"\n📁 Input directory: {CONFIG['input_dir']}")
print(f"📁 Output directory: {CONFIG['output_dir']}")

#%% Đọc dữ liệu từ file annotation
def parse_annotation_file(file_path):
    """Parse file annotation YOLO để lấy dữ liệu"""
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        return None, None, None
    
    # Đọc thời gian ngã
    fall_start = int(lines[0].strip())
    fall_end = int(lines[1].strip())
    
    # Đọc dữ liệu frames
    for line in lines[2:]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            frame_num = int(parts[0])
            label = int(parts[1])
            bbox = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
            
            # Parse keypoints
            keypoints = []
            for i in range(6, len(parts)):
                if 'kp' in parts[i]:
                    kp_parts = parts[i].split(':')
                    if len(kp_parts) == 4:
                        x = float(kp_parts[1])
                        y = float(kp_parts[2])
                        conf = float(kp_parts[3])
                        keypoints.extend([x, y, conf])
            
            # Đảm bảo có đủ 17 keypoints
            while len(keypoints) < 51:
                keypoints.extend([0, 0, 0])
            keypoints = keypoints[:51]
            
            data.append({
                'frame': frame_num,
                'label': label,
                'bbox': bbox,
                'keypoints': np.array(keypoints).reshape(17, 3)
            })
    
    return data, fall_start, fall_end

#%% 1. GEOMETRIC FEATURES (Đặc trưng hình học)
def extract_geometric_features(keypoints, bbox):
    """Trích xuất đặc trưng hình học từ pose"""
    features = {}
    
    # Lọc keypoints có confidence > threshold
    valid_kps = keypoints[keypoints[:, 2] > CONFIG['min_confidence']]
    
    if len(valid_kps) == 0:
        return {f'geo_{k}': 0 for k in ['height', 'width', 'aspect_ratio', 'area', 'center_x', 'center_y',
                                       'head_torso_ratio', 'arm_span', 'leg_length', 'body_symmetry']}
    
    # Bbox features
    x1, y1, x2, y2 = bbox
    features['geo_height'] = y2 - y1
    features['geo_width'] = x2 - x1
    features['geo_aspect_ratio'] = (y2 - y1) / max(x2 - x1, 1)
    features['geo_area'] = (x2 - x1) * (y2 - y1)
    features['geo_center_x'] = (x1 + x2) / 2
    features['geo_center_y'] = (y1 + y2) / 2
    
    # Pose geometric features
    # Head-torso ratio
    head_kps = keypoints[KEYPOINT_GROUPS['head']]
    torso_kps = keypoints[KEYPOINT_GROUPS['torso']]
    
    head_valid = head_kps[head_kps[:, 2] > CONFIG['min_confidence']]
    torso_valid = torso_kps[torso_kps[:, 2] > CONFIG['min_confidence']]
    
    if len(head_valid) > 0 and len(torso_valid) > 0:
        head_height = np.max(head_valid[:, 1]) - np.min(head_valid[:, 1])
        torso_height = np.max(torso_valid[:, 1]) - np.min(torso_valid[:, 1])
        features['geo_head_torso_ratio'] = head_height / max(torso_height, 1)
    else:
        features['geo_head_torso_ratio'] = 0
    
    # Arm span
    left_shoulder = keypoints[5] if keypoints[5, 2] > CONFIG['min_confidence'] else None
    right_shoulder = keypoints[6] if keypoints[6, 2] > CONFIG['min_confidence'] else None
    left_wrist = keypoints[9] if keypoints[9, 2] > CONFIG['min_confidence'] else None
    right_wrist = keypoints[10] if keypoints[10, 2] > CONFIG['min_confidence'] else None
    
    if left_wrist is not None and right_wrist is not None:
        features['geo_arm_span'] = euclidean(left_wrist[:2], right_wrist[:2])
    else:
        features['geo_arm_span'] = 0
    
    # Leg length
    left_hip = keypoints[11] if keypoints[11, 2] > CONFIG['min_confidence'] else None
    left_ankle = keypoints[15] if keypoints[15, 2] > CONFIG['min_confidence'] else None
    
    if left_hip is not None and left_ankle is not None:
        features['geo_leg_length'] = euclidean(left_hip[:2], left_ankle[:2])
    else:
        features['geo_leg_length'] = 0
    
    # Body symmetry
    left_points = keypoints[[1, 3, 5, 7, 9, 11, 13, 15]]  # Left side
    right_points = keypoints[[2, 4, 6, 8, 10, 12, 14, 16]]  # Right side
    
    left_valid = left_points[left_points[:, 2] > CONFIG['min_confidence']]
    right_valid = right_points[right_points[:, 2] > CONFIG['min_confidence']]
    
    if len(left_valid) > 0 and len(right_valid) > 0:
        # Tính độ đối xứng dựa trên khoảng cách từ trung tâm
        center_x = np.mean(valid_kps[:, 0])
        left_dist = np.mean(np.abs(left_valid[:, 0] - center_x))
        right_dist = np.mean(np.abs(right_valid[:, 0] - center_x))
        features['geo_body_symmetry'] = 1 - abs(left_dist - right_dist) / max(left_dist + right_dist, 1)
    else:
        features['geo_body_symmetry'] = 0
    
    return features

#%% 2. MOTION FEATURES (Đặc trưng chuyển động)
def extract_motion_features(keypoints_sequence):
    """Trích xuất đặc trưng chuyển động từ chuỗi keypoints"""
    features = {}
    
    if len(keypoints_sequence) < 2:
        return {f'motion_{k}': 0 for k in ['velocity_mean', 'velocity_std', 'acceleration_mean', 
                                          'acceleration_std', 'displacement', 'direction_change']}
    
    # Tính velocity và acceleration cho từng keypoint
    velocities = []
    accelerations = []
    
    for i in range(len(keypoints_sequence) - 1):
        kp1 = keypoints_sequence[i]
        kp2 = keypoints_sequence[i + 1]
        
        # Chỉ tính cho keypoints có confidence > threshold
        for j in range(17):
            if kp1[j, 2] > CONFIG['min_confidence'] and kp2[j, 2] > CONFIG['min_confidence']:
                vel = euclidean(kp1[j, :2], kp2[j, :2])
                velocities.append(vel)
    
    # Tính acceleration
    for i in range(len(velocities) - 1):
        acc = abs(velocities[i + 1] - velocities[i])
        accelerations.append(acc)
    
    # Motion features
    features['motion_velocity_mean'] = np.mean(velocities) if velocities else 0
    features['motion_velocity_std'] = np.std(velocities) if velocities else 0
    features['motion_acceleration_mean'] = np.mean(accelerations) if accelerations else 0
    features['motion_acceleration_std'] = np.std(accelerations) if accelerations else 0
    
    # Total displacement
    if len(keypoints_sequence) > 1:
        first_frame = keypoints_sequence[0]
        last_frame = keypoints_sequence[-1]
        
        displacements = []
        for j in range(17):
            if (first_frame[j, 2] > CONFIG['min_confidence'] and 
                last_frame[j, 2] > CONFIG['min_confidence']):
                disp = euclidean(first_frame[j, :2], last_frame[j, :2])
                displacements.append(disp)
        
        features['motion_displacement'] = np.mean(displacements) if displacements else 0
    else:
        features['motion_displacement'] = 0
    
    # Direction change (số lần thay đổi hướng)
    direction_changes = 0
    if len(keypoints_sequence) > 2:
        for j in range(17):
            directions = []
            for i in range(len(keypoints_sequence) - 1):
                kp1 = keypoints_sequence[i]
                kp2 = keypoints_sequence[i + 1]
                
                if kp1[j, 2] > CONFIG['min_confidence'] and kp2[j, 2] > CONFIG['min_confidence']:
                    dx = kp2[j, 0] - kp1[j, 0]
                    dy = kp2[j, 1] - kp1[j, 1]
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
            
            # Đếm số lần thay đổi hướng
            for i in range(len(directions) - 1):
                angle_diff = abs(directions[i + 1] - directions[i])
                if angle_diff > np.pi / 4:  # Thay đổi > 45 độ
                    direction_changes += 1
    
    features['motion_direction_change'] = direction_changes
    
    return features

#%% 3. POSE FEATURES (Đặc trưng tư thế)
def extract_pose_features(keypoints):
    """Trích xuất đặc trưng tư thế"""
    features = {}
    
    # Angle features
    angles = calculate_body_angles(keypoints)
    for angle_name, angle_value in angles.items():
        features[f'pose_angle_{angle_name}'] = angle_value
    
    # Pose stability (độ ổn định tư thế)
    features['pose_stability'] = calculate_pose_stability(keypoints)
    
    # Body orientation
    features['pose_orientation'] = calculate_body_orientation(keypoints)
    
    # Limb positions relative to body center
    limb_positions = calculate_limb_positions(keypoints)
    for limb_name, position in limb_positions.items():
        features[f'pose_limb_{limb_name}'] = position
    
    return features

def calculate_body_angles(keypoints):
    """Tính các góc cơ thể quan trọng"""
    angles = {}
    
    # Shoulder-hip angle (góc nghiêng cơ thể)
    left_shoulder = keypoints[5] if keypoints[5, 2] > CONFIG['min_confidence'] else None
    right_shoulder = keypoints[6] if keypoints[6, 2] > CONFIG['min_confidence'] else None
    left_hip = keypoints[11] if keypoints[11, 2] > CONFIG['min_confidence'] else None
    right_hip = keypoints[12] if keypoints[12, 2] > CONFIG['min_confidence'] else None
    
    if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
        # Tính góc giữa đường vai và đường hông
        shoulder_vector = right_shoulder[:2] - left_shoulder[:2]
        hip_vector = right_hip[:2] - left_hip[:2]
        
        shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        hip_angle = np.arctan2(hip_vector[1], hip_vector[0])
        
        angles['torso_tilt'] = abs(shoulder_angle - hip_angle)
    else:
        angles['torso_tilt'] = 0
    
    # Knee angles
    for side, (hip_idx, knee_idx, ankle_idx) in [('left', (11, 13, 15)), ('right', (12, 14, 16))]:
        hip = keypoints[hip_idx] if keypoints[hip_idx, 2] > CONFIG['min_confidence'] else None
        knee = keypoints[knee_idx] if keypoints[knee_idx, 2] > CONFIG['min_confidence'] else None
        ankle = keypoints[ankle_idx] if keypoints[ankle_idx, 2] > CONFIG['min_confidence'] else None
        
        if all(p is not None for p in [hip, knee, ankle]):
            # Tính góc tại đầu gối
            v1 = hip[:2] - knee[:2]
            v2 = ankle[:2] - knee[:2]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles[f'{side}_knee'] = angle
        else:
            angles[f'{side}_knee'] = 0
    
    # Elbow angles
    for side, (shoulder_idx, elbow_idx, wrist_idx) in [('left', (5, 7, 9)), ('right', (6, 8, 10))]:
        shoulder = keypoints[shoulder_idx] if keypoints[shoulder_idx, 2] > CONFIG['min_confidence'] else None
        elbow = keypoints[elbow_idx] if keypoints[elbow_idx, 2] > CONFIG['min_confidence'] else None
        wrist = keypoints[wrist_idx] if keypoints[wrist_idx, 2] > CONFIG['min_confidence'] else None
        
        if all(p is not None for p in [shoulder, elbow, wrist]):
            v1 = shoulder[:2] - elbow[:2]
            v2 = wrist[:2] - elbow[:2]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles[f'{side}_elbow'] = angle
        else:
            angles[f'{side}_elbow'] = 0
    
    return angles

def calculate_pose_stability(keypoints):
    """Tính độ ổn định tư thế"""
    valid_kps = keypoints[keypoints[:, 2] > CONFIG['min_confidence']]
    
    if len(valid_kps) == 0:
        return 0
    
    # Tính độ phân tán của các keypoints
    center_x = np.mean(valid_kps[:, 0])
    center_y = np.mean(valid_kps[:, 1])
    
    distances = [euclidean([kp[0], kp[1]], [center_x, center_y]) for kp in valid_kps]
    stability = 1 / (1 + np.std(distances))  # Càng ít phân tán càng ổn định
    
    return stability

def calculate_body_orientation(keypoints):
    """Tính hướng cơ thể (đứng, nằm, nghiêng)"""
    # Sử dụng vector từ hông đến vai để xác định orientation
    left_shoulder = keypoints[5] if keypoints[5, 2] > CONFIG['min_confidence'] else None
    right_shoulder = keypoints[6] if keypoints[6, 2] > CONFIG['min_confidence'] else None
    left_hip = keypoints[11] if keypoints[11, 2] > CONFIG['min_confidence'] else None
    right_hip = keypoints[12] if keypoints[12, 2] > CONFIG['min_confidence'] else None
    
    if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip_center = (left_hip[:2] + right_hip[:2]) / 2
        
        # Vector từ hông đến vai
        torso_vector = shoulder_center - hip_center
        
        # Góc với trục dọc (0 = đứng thẳng, π/2 = nằm ngang)
        orientation = np.arctan2(abs(torso_vector[0]), abs(torso_vector[1]))
        
        return orientation
    else:
        return 0

def calculate_limb_positions(keypoints):
    """Tính vị trí các chi so với trung tâm cơ thể"""
    positions = {}
    
    # Tính trung tâm cơ thể
    torso_kps = keypoints[KEYPOINT_GROUPS['torso']]
    valid_torso = torso_kps[torso_kps[:, 2] > CONFIG['min_confidence']]
    
    if len(valid_torso) == 0:
        return {f'{limb}_{coord}': 0 for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot'] 
                for coord in ['x', 'y', 'dist']}
    
    center_x = np.mean(valid_torso[:, 0])
    center_y = np.mean(valid_torso[:, 1])
    
    # Vị trí tay và chân
    limb_indices = {
        'left_hand': 9,   # left_wrist
        'right_hand': 10, # right_wrist
        'left_foot': 15,  # left_ankle
        'right_foot': 16  # right_ankle
    }
    
    for limb_name, idx in limb_indices.items():
        if keypoints[idx, 2] > CONFIG['min_confidence']:
            x_rel = keypoints[idx, 0] - center_x
            y_rel = keypoints[idx, 1] - center_y
            dist = euclidean([keypoints[idx, 0], keypoints[idx, 1]], [center_x, center_y])
            
            positions[f'{limb_name}_x'] = x_rel
            positions[f'{limb_name}_y'] = y_rel
            positions[f'{limb_name}_dist'] = dist
        else:
            positions[f'{limb_name}_x'] = 0
            positions[f'{limb_name}_y'] = 0
            positions[f'{limb_name}_dist'] = 0
    
    return positions

#%% 4. TEMPORAL FEATURES (Đặc trưng thời gian)
def extract_temporal_features(data_sequence):
    """Trích xuất đặc trưng thời gian từ chuỗi dữ liệu"""
    features = {}
    
    if len(data_sequence) < 2:
        return {f'temporal_{k}': 0 for k in ['duration', 'frequency', 'periodicity', 'trend']}
    
    # Duration of movement
    features['temporal_duration'] = len(data_sequence)
    
    # Frequency analysis
    # Lấy trajectory của một keypoint quan trọng (nose)
    nose_trajectory = []
    for frame_data in data_sequence:
        if frame_data['keypoints'][0, 2] > CONFIG['min_confidence']:
            nose_trajectory.append(frame_data['keypoints'][0, 1])  # Y coordinate
        else:
            nose_trajectory.append(0)
    
    if len(nose_trajectory) > 1:
        # FFT để phân tích tần số
        fft = np.fft.fft(nose_trajectory)
        freqs = np.fft.fftfreq(len(nose_trajectory))
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        features['temporal_frequency'] = abs(freqs[dominant_freq_idx])
        
        # Periodicity (độ tuần hoàn)
        autocorr = np.correlate(nose_trajectory, nose_trajectory, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Tìm peaks trong autocorrelation
        peaks, _ = find_peaks(autocorr[1:], height=np.max(autocorr) * 0.3)
        features['temporal_periodicity'] = len(peaks)
        
        # Trend (xu hướng tăng/giảm)
        if len(nose_trajectory) > 2:
            slope, _, _, _, _ = stats.linregress(range(len(nose_trajectory)), nose_trajectory)
            features['temporal_trend'] = abs(slope)
        else:
            features['temporal_trend'] = 0
    else:
        features['temporal_frequency'] = 0
        features['temporal_periodicity'] = 0
        features['temporal_trend'] = 0
    
    return features

#%% 5. STATISTICAL FEATURES (Đặc trưng thống kê)
def extract_statistical_features(data_sequence):
    """Trích xuất đặc trưng thống kê"""
    features = {}
    
    if len(data_sequence) == 0:
        return {f'stat_{k}': 0 for k in ['mean', 'std', 'skewness', 'kurtosis', 'entropy']}
    
    # Tập hợp tất cả keypoint coordinates
    all_coords = []
    confidences = []
    
    for frame_data in data_sequence:
        for kp in frame_data['keypoints']:
            if kp[2] > CONFIG['min_confidence']:
                all_coords.extend([kp[0], kp[1]])
                confidences.append(kp[2])
    
    if len(all_coords) > 0:
        # Statistical moments
        features['stat_mean'] = np.mean(all_coords)
        features['stat_std'] = np.std(all_coords)
        features['stat_skewness'] = stats.skew(all_coords)
        features['stat_kurtosis'] = stats.kurtosis(all_coords)
        
        # Entropy (độ hỗn loạn)
        hist, _ = np.histogram(all_coords, bins=20)
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        features['stat_entropy'] = entropy
        
        # Confidence statistics
        features['stat_conf_mean'] = np.mean(confidences)
        features['stat_conf_std'] = np.std(confidences)
    else:
        for key in ['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'conf_mean', 'conf_std']:
            features[f'stat_{key}'] = 0
    
    return features

#%% 6. FREQUENCY FEATURES (Đặc trưng tần số)
def extract_frequency_features(data_sequence):
    """Trích xuất đặc trưng tần số"""
    features = {}
    
    if len(data_sequence) < 4:
        return {f'freq_{k}': 0 for k in ['low', 'mid', 'high', 'spectral_centroid', 'spectral_rolloff']}
    
    # Lấy trajectory của center of mass
    com_trajectory = []
    for frame_data in data_sequence:
        valid_kps = frame_data['keypoints'][frame_data['keypoints'][:, 2] > CONFIG['min_confidence']]
        if len(valid_kps) > 0:
            com_x = np.mean(valid_kps[:, 0])
            com_y = np.mean(valid_kps[:, 1])
            com_trajectory.append([com_x, com_y])
        else:
            com_trajectory.append([0, 0])
    
    com_trajectory = np.array(com_trajectory)
    
    # FFT analysis cho X và Y
    for axis, axis_name in [(0, 'x'), (1, 'y')]:
        signal = com_trajectory[:, axis]
        
        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        power_spectrum = np.abs(fft) ** 2
        
        # Frequency bands
        low_freq_power = np.sum(power_spectrum[(freqs >= 0) & (freqs < 0.1)])
        mid_freq_power = np.sum(power_spectrum[(freqs >= 0.1) & (freqs < 0.3)])
        high_freq_power = np.sum(power_spectrum[(freqs >= 0.3) & (freqs < 0.5)])
        
        total_power = np.sum(power_spectrum[freqs >= 0])
        
        features[f'freq_low_{axis_name}'] = low_freq_power / (total_power + 1e-8)
        features[f'freq_mid_{axis_name}'] = mid_freq_power / (total_power + 1e-8)
        features[f'freq_high_{axis_name}'] = high_freq_power / (total_power + 1e-8)
        
        # Spectral centroid
        if total_power > 0:
            spectral_centroid = np.sum(freqs[freqs >= 0] * power_spectrum[freqs >= 0]) / total_power
            features[f'freq_spectral_centroid_{axis_name}'] = spectral_centroid
        else:
            features[f'freq_spectral_centroid_{axis_name}'] = 0
    
    return features

#%% Main feature extraction function
def extract_all_features(data_sequence):
    """Trích xuất tất cả đặc trưng từ một chuỗi dữ liệu"""
    all_features = {}
    
    if len(data_sequence) == 0:
        return all_features
    
    # 1. Geometric features (từ frame cuối)
    last_frame = data_sequence[-1]
    geo_features = extract_geometric_features(last_frame['keypoints'], last_frame['bbox'])
    all_features.update(geo_features)
    
    # 2. Motion features
    keypoints_sequence = [frame['keypoints'] for frame in data_sequence]
    motion_features = extract_motion_features(keypoints_sequence)
    all_features.update(motion_features)
    
    # 3. Pose features (trung bình từ tất cả frames)
    pose_features_list = []
    for frame in data_sequence:
        pose_feat = extract_pose_features(frame['keypoints'])
        pose_features_list.append(pose_feat)
    
    # Tính trung bình pose features
    if pose_features_list:
        pose_keys = pose_features_list[0].keys()
        for key in pose_keys:
            values = [pf[key] for pf in pose_features_list if key in pf]
            all_features[key] = np.mean(values) if values else 0
    
    # 4. Temporal features
    temporal_features = extract_temporal_features(data_sequence)
    all_features.update(temporal_features)
    
    # 5. Statistical features
    stat_features = extract_statistical_features(data_sequence)
    all_features.update(stat_features)
    
    # 6. Frequency features
    freq_features = extract_frequency_features(data_sequence)
    all_features.update(freq_features)
    
    return all_features

#%% Process all files
def process_all_files():
    """Xử lý tất cả files và trích xuất đặc trưng"""
    
    print(f"\n🔄 Bắt đầu trích xuất đặc trưng...")
    
    # Tìm tất cả files
    all_files = []
    for dataset_dir in os.listdir(CONFIG['input_dir']):
        dataset_path = os.path.join(CONFIG['input_dir'], dataset_dir)
        if os.path.isdir(dataset_path):
            files = glob.glob(os.path.join(dataset_path, "*_yolo_pose.txt"))
            all_files.extend(files)
    
    print(f"📊 Tìm thấy {len(all_files)} files")
    
    # Tạo output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    all_features_data = []
    
    for file_path in tqdm(all_files, desc="Extracting features"):
        # Parse file
        data, fall_start, fall_end = parse_annotation_file(file_path)
        
        if data is None:
            continue
        
        # Sliding window approach
        window_size = CONFIG['window_size']
        overlap = CONFIG['overlap']
        step = window_size - overlap
        
        for start_idx in range(0, len(data) - window_size + 1, step):
            end_idx = start_idx + window_size
            window_data = data[start_idx:end_idx]
            
            # Xác định label cho window
            window_frames = [d['frame'] for d in window_data]
            falling_frames = [f for f in window_frames if fall_start <= f <= fall_end]
            
            # Label: 1 nếu > 50% frames trong window là falling
            label = 1 if len(falling_frames) > window_size // 2 else 0
            
            # Trích xuất đặc trưng
            features = extract_all_features(window_data)
            
            # Thêm metadata
            features['file_name'] = os.path.basename(file_path)
            features['window_start'] = start_idx
            features['window_end'] = end_idx
            features['label'] = label
            features['fall_ratio'] = len(falling_frames) / window_size
            
            all_features_data.append(features)
    
    # Chuyển thành DataFrame
    df = pd.DataFrame(all_features_data)
    
    # Lưu kết quả
    output_file = os.path.join(CONFIG['output_dir'], 'extracted_features.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Hoàn thành trích xuất đặc trưng!")
    print(f"📊 Tổng số windows: {len(df)}")
    print(f"📊 Normal windows: {len(df[df['label'] == 0])}")
    print(f"📊 Falling windows: {len(df[df['label'] == 1])}")
    print(f"📊 Số đặc trưng: {len(df.columns) - 5}")  # Trừ metadata columns
    print(f"💾 Saved: {output_file}")
    
    # Thống kê đặc trưng
    feature_columns = [col for col in df.columns if col not in ['file_name', 'window_start', 'window_end', 'label', 'fall_ratio']]
    
    print(f"\n📋 Loại đặc trưng:")
    feature_types = {}
    for col in feature_columns:
        prefix = col.split('_')[0]
        if prefix not in feature_types:
            feature_types[prefix] = 0
        feature_types[prefix] += 1
    
    for feat_type, count in feature_types.items():
        print(f"   {feat_type}: {count} features")
    
    return df

if __name__ == "__main__":
    # Chạy trích xuất đặc trưng
    features_df = process_all_files()
    
    print(f"\n🎉 Feature extraction hoàn thành!")
    print(f"📁 Kết quả lưu tại: {CONFIG['output_dir']}") 