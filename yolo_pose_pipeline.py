#%% YOLO Pose Detection Pipeline để tạo Dataset mới
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
import json
from tqdm import tqdm
from pathlib import Path

print("🚀 YOLO POSE DETECTION PIPELINE")
print("=" * 50)
print("📋 Mục tiêu:")
print("   ✅ Sử dụng YOLO detect người và pose")
print("   ✅ Đọc nhãn cũ lấy thời gian ngã")
print("   ✅ Tạo file nhãn mới với YOLO detection")
print("   ✅ Xử lý tất cả video trong dataset")

#%% Cấu hình đường dẫn
CONFIG = {
    # Đường dẫn dữ liệu - GIỮ NGUYÊN THEO YÊU CẦU
    'datasets': [
        {
            'name': 'Coffee_room_01',
            'video_path': 'khoaminh/data/Coffee_room_01/Coffee_room_01/Videos',
            'annotation_path': 'khoaminh/data/Coffee_room_01/Coffee_room_01/Annotation_files_processed'
        },
        {
            'name': 'Coffee_room_02', 
            'video_path': 'khoaminh/data/Coffee_room_02/Coffee_room_02/Videos',
            'annotation_path': 'khoaminh/data/Coffee_room_02/Coffee_room_02/Annotation_files_processed'
        },
        {
            'name': 'Home_01',
            'video_path': 'khoaminh/data/Home_01/Home_01/Videos', 
            'annotation_path': 'khoaminh/data/Home_01/Home_01/Annotation_files_processed'
        },
        {
            'name': 'Home_02',
            'video_path': 'khoaminh/data/Home_02/Home_02/Videos',
            'annotation_path': 'khoaminh/data/Home_02/Home_02/Annotation_files_processed'
        }
    ],
    
    # Đường dẫn output
    'output_dir': 'new_dataset',
    
    # YOLO model
    'yolo_model': 'yolo11x-pose.pt',  # YOLOv8 Pose model
    
    # Tham số detection
    'confidence_threshold': 0.5,
    'person_class_id': 0,  # Class ID cho person trong COCO
    
    # Tham số khác
    'max_frames_per_video': None,  # None = xử lý tất cả frames
}

print(f"\n📁 Sẽ xử lý {len(CONFIG['datasets'])} datasets:")
for dataset in CONFIG['datasets']:
    print(f"   📂 {dataset['name']}")

#%% Load YOLO model
def load_yolo_model():
    """Load YOLO Pose model"""
    print(f"\n🤖 Loading YOLO model: {CONFIG['yolo_model']}")
    try:
        model = YOLO(CONFIG['yolo_model'])
        print("✅ YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Lỗi load YOLO model: {e}")
        return None

#%% Đọc nhãn cũ để lấy thời gian ngã
def read_old_annotation(annotation_file):
    """Đọc file annotation cũ để lấy thời gian ngã"""
    try:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 2:
            print(f"❌ File annotation không đủ dữ liệu: {annotation_file}")
            return None, None
            
        # Dòng đầu tiên: frame bắt đầu ngã
        fall_start = int(lines[0].strip())
        # Dòng thứ hai: frame kết thúc ngã  
        fall_end = int(lines[1].strip())
        
        print(f"📊 Thời gian ngã: frame {fall_start} -> {fall_end}")
        return fall_start, fall_end
        
    except Exception as e:
        print(f"❌ Lỗi đọc annotation {annotation_file}: {e}")
        return None, None

#%% YOLO Detection và Pose Estimation
def detect_person_pose(model, frame):
    """Detect người và pose keypoints bằng YOLO"""
    try:
        # Run YOLO inference
        results = model(frame, conf=CONFIG['confidence_threshold'], verbose=False)
        
        detections = []
        
        for result in results:
            # Lấy boxes và keypoints
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.data.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Chỉ lấy detection của person (class 0)
                    if int(box[5]) == CONFIG['person_class_id']:
                        # Bbox coordinates
                        x1, y1, x2, y2 = box[:4]
                        confidence = box[4]
                        
                        # Keypoints (17 keypoints, mỗi keypoint có x, y, confidence)
                        if i < len(keypoints):
                            kp = keypoints[i]  # Shape: (17, 3)
                            
                            detection = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence),
                                'keypoints': kp.flatten().tolist()  # Flatten to 51 values
                            }
                            detections.append(detection)
        
        return detections
        
    except Exception as e:
        print(f"❌ Lỗi YOLO detection: {e}")
        return []

#%% Tìm video và annotation tương ứng
def find_matching_files(dataset_config):
    """Tìm các cặp video-annotation tương ứng"""
    video_dir = dataset_config['video_path']
    annotation_dir = dataset_config['annotation_path']
    
    print(f"\n🔍 Tìm kiếm files trong {dataset_config['name']}:")
    print(f"   📹 Video dir: {video_dir}")
    print(f"   📝 Annotation dir: {annotation_dir}")
    
    # Tìm video files
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    # Tìm annotation files
    annotation_files = glob.glob(os.path.join(annotation_dir, "*_with_pose.txt"))
    
    print(f"   📊 Tìm thấy: {len(video_files)} videos, {len(annotation_files)} annotations")
    
    # Ghép video với annotation
    matched_pairs = []
    for ann_file in annotation_files:
        ann_basename = os.path.basename(ann_file)
        # Loại bỏ "_with_pose.txt" để lấy tên video
        video_name_base = ann_basename.replace('_with_pose.txt', '')
        
        # Thử các tên video có thể có
        possible_video_names = [
            video_name_base.replace('_', ' ') + '.avi',
            video_name_base.replace('_', ' ') + '.mp4',
            video_name_base + '.avi',
            video_name_base + '.mp4',
        ]
        
        matching_video = None
        for video_file in video_files:
            video_basename = os.path.basename(video_file)
            if video_basename in possible_video_names:
                matching_video = video_file
                break
        
        if matching_video:
            matched_pairs.append({
                'video_path': matching_video,
                'annotation_path': ann_file,
                'video_name': video_name_base
            })
            print(f"   ✅ Matched: {os.path.basename(matching_video)} <-> {ann_basename}")
        else:
            print(f"   ❌ No video found for: {ann_basename}")
    
    print(f"   📋 Tổng cộng: {len(matched_pairs)} cặp video-annotation")
    return matched_pairs

#%% Xử lý một video
def process_video(model, video_info, dataset_name, output_dir):
    """Xử lý một video để tạo nhãn mới"""
    video_path = video_info['video_path']
    annotation_path = video_info['annotation_path']
    video_name = video_info['video_name']
    
    print(f"\n🎬 Xử lý video: {os.path.basename(video_path)}")
    
    # Đọc thời gian ngã từ annotation cũ
    fall_start, fall_end = read_old_annotation(annotation_path)
    if fall_start is None or fall_end is None:
        print(f"❌ Không thể đọc thời gian ngã, bỏ qua video này")
        return False
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # Tạo thư mục output
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Tên file output
    output_file = os.path.join(dataset_output_dir, f"{video_name}_yolo_pose.txt")
    
    # Xử lý từng frame
    frame_detections = {}
    frame_count = 0
    
    max_frames = CONFIG['max_frames_per_video'] or total_frames
    
    with tqdm(total=min(total_frames, max_frames), desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            # YOLO detection
            detections = detect_person_pose(model, frame)
            
            if detections:
                # Lấy detection tốt nhất (confidence cao nhất)
                best_detection = max(detections, key=lambda x: x['confidence'])
                frame_detections[frame_count] = best_detection
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    # Ghi file nhãn mới
    write_new_annotation(output_file, frame_detections, fall_start, fall_end, total_frames)
    
    print(f"✅ Hoàn thành: {len(frame_detections)}/{total_frames} frames có detection")
    print(f"📊 Tất cả {total_frames} frames đã được ghi để khớp với data cũ")
    print(f"💾 Saved: {output_file}")
    
    return True

#%% Ghi file annotation mới
def write_new_annotation(output_file, frame_detections, fall_start, fall_end, total_frames):
    """Ghi file annotation mới với format chuẩn"""
    
    with open(output_file, 'w') as f:
        # Dòng 1: Frame bắt đầu ngã
        f.write(f"{fall_start}\n")
        
        # Dòng 2: Frame kết thúc ngã
        f.write(f"{fall_end}\n")
        
        # Các dòng tiếp theo: GHI TẤT CẢ FRAMES để khớp với data cũ
        for frame_num in range(total_frames):
            # Label đơn giản: chỉ phân biệt ngã hay không ngã
            if fall_start <= frame_num <= fall_end:
                label = 1  # Falling (đang ngã)
            else:
                label = 0  # Normal (không ngã)
            
            if frame_num in frame_detections:
                # Frame có detection
                detection = frame_detections[frame_num]
                
                # Bbox coordinates
                x1, y1, x2, y2 = detection['bbox']
                
                # Keypoints (51 values: 17 keypoints * 3 (x, y, conf))
                keypoints = detection['keypoints']
                
                # Format keypoints as string
                kp_str = ""
                for i in range(0, len(keypoints), 3):
                    if i + 2 < len(keypoints):
                        x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                        kp_str += f"kp{i//3}:{x:.2f}:{y:.2f}:{conf:.2f},"
                
                # Loại bỏ dấu phẩy cuối
                kp_str = kp_str.rstrip(',')
                
                # Ghi dòng với detection
                line = f"{frame_num},{label},{int(x1)},{int(y1)},{int(x2)},{int(y2)},{kp_str}\n"
                f.write(line)
            else:
                # Frame KHÔNG có detection - ghi với bbox và keypoints = 0
                # Tạo keypoints rỗng (17 keypoints * 3 = 51 values, tất cả = 0)
                empty_kp_str = ",".join([f"kp{i}:0.00:0.00:0.00" for i in range(17)])
                
                # Ghi dòng với bbox = 0 và keypoints = 0
                line = f"{frame_num},{label},0,0,0,0,{empty_kp_str}\n"
                f.write(line)

#%% Main pipeline
def main():
    """Main pipeline function"""
    print("\n🚀 Bắt đầu YOLO Pose Detection Pipeline...")
    
    # Load YOLO model
    model = load_yolo_model()
    if model is None:
        print("❌ Không thể load YOLO model, dừng pipeline")
        return
    
    # Tạo thư mục output
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Xử lý từng dataset
    total_processed = 0
    total_failed = 0
    
    for dataset_config in CONFIG['datasets']:
        print(f"\n{'='*60}")
        print(f"📂 Xử lý dataset: {dataset_config['name']}")
        print(f"{'='*60}")
        
        # Tìm các cặp video-annotation
        matched_pairs = find_matching_files(dataset_config)
        
        if not matched_pairs:
            print(f"❌ Không tìm thấy cặp video-annotation nào trong {dataset_config['name']}")
            continue
        
        # Xử lý từng video
        for video_info in matched_pairs:
            try:
                success = process_video(model, video_info, dataset_config['name'], output_dir)
                if success:
                    total_processed += 1
                else:
                    total_failed += 1
            except Exception as e:
                print(f"❌ Lỗi xử lý video {video_info['video_name']}: {e}")
                total_failed += 1
    
    # Tổng kết
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE HOÀN THÀNH!")
    print(f"{'='*60}")
    print(f"✅ Thành công: {total_processed} videos")
    print(f"❌ Thất bại: {total_failed} videos")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Format file mới: frame_start, frame_end, frame_data...")
    print(f"📋 Label mapping: 0=Normal, 1=Falling")
    print(f"📋 Chỉ lấy thời gian ngã từ data cũ, logic phân loại đơn giản!")
    print(f"📋 GHI TẤT CẢ FRAMES (kể cả không có detection) để khớp với data cũ!")

if __name__ == "__main__":
    main() 