import cv2
import os
import numpy as np # Mặc dù không ghép video nữa, vẫn có thể hữu ích cho các thao tác khác

# ----- CẤU HÌNH ĐƯỜNG DẪN -----
# Hãy đảm bảo các đường dẫn này là chính xác!
annotation_file_path = '/workspace/khoaminh/data/Coffee_room_02/Coffee_room_02/Annotations_files/video (53).txt'
video_file_path = '/workspace/khoaminh/data/Coffee_room_02/Coffee_room_02/Videos/video (53).avi'

# Thư mục và tên tệp video output cuối cùng
output_dir = '/workspace/khoaminh/data/Coffee_room_02/Coffee_room_02/Videos/processed_fall_detection/'
final_output_video_name = 'video_1_processed_CORRECTED.avi'
output_video_path = os.path.join(output_dir, final_output_video_name)
# -------------------------------

# --- Tạo thư mục output nếu chưa có ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Đã tạo thư mục output: {output_dir}")

def parse_annotations_topleft_bottomright(file_path):
    """
    Đọc và phân tích tệp chú thích theo định dạng:
    frame_num, label, x1, y1, x2, y2
    """
    annotations = {}
    fall_start_frame = -1
    fall_end_frame = -1
    try:
        with open(file_path, 'r') as f:
            fall_start_frame = int(f.readline().strip())
            fall_end_frame = int(f.readline().strip())
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 6:
                    frame_num = int(parts[0])
                    annotations[frame_num] = {
                        'label': int(parts[1]),
                        'x1': int(parts[2]), 
                        'y1': int(parts[3]), 
                        'x2': int(parts[4]), 
                        'y2': int(parts[5])  
                    }
        print(f"Đọc tệp chú thích '{file_path}' (kiểu topleft_bottomright) thành công.")
        print(f"  Khung hình bắt đầu ngã: {fall_start_frame}")
        print(f"  Khung hình kết thúc ngã: {fall_end_frame}")
        return annotations, fall_start_frame, fall_end_frame
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp chú thích tại: {file_path}")
    except Exception as e:
        print(f"LỖI khi đọc tệp chú thích '{file_path}': {e}")
    return None, -1, -1

def get_video_properties(video_path):
    """Lấy kích thước (width, height) và FPS của video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video '{video_path}' để lấy properties.")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if width == 0 or height == 0 or fps <= 0: # FPS không thể là 0 hoặc âm
        print(f"LỖI: Thông tin video không hợp lệ (W:{width}, H:{height}, FPS:{fps}) từ '{video_path}'.")
        return None, None
        
    return (width, height), fps

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Phân tích chú thích
    annotations, fall_start, fall_end = parse_annotations_topleft_bottomright(annotation_file_path)
    if annotations is None:
        print("Không thể xử lý do lỗi đọc chú thích. Thoát.")
        exit()

    # 2. Lấy thông tin video gốc (kích thước, fps)
    original_video_size, original_video_fps = get_video_properties(video_file_path)
    if original_video_size is None:
        print("Không thể xử lý do lỗi đọc video gốc. Thoát.")
        exit()
    
    video_width, video_height = original_video_size

    # 3. Xử lý video với định dạng bounding box đã xác định
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video gốc '{video_file_path}'.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec cho file .avi
    out = cv2.VideoWriter(output_video_path, fourcc, original_video_fps, original_video_size)
    if not out.isOpened():
        print(f"LỖI: Không thể tạo video output '{output_video_path}'.")
        cap.release()
        exit()

    print(f"\nĐang xử lý video với bounding box kiểu 'topleft_bottomright'...")
    print(f"Video thành phẩm sẽ được lưu tại: '{output_video_path}'")
    
    frame_num_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Kết thúc video hoặc lỗi đọc frame
        frame_num_counter += 1
        
        processed_frame = frame.copy() # Làm việc trên bản sao của frame
        
        is_falling_period = (fall_start <= frame_num_counter <= fall_end)
        
        if frame_num_counter in annotations:
            ann = annotations[frame_num_counter]
            x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            label = ann['label']
            
            # Kiểm tra tính hợp lệ của bounding box (x2 > x1 và y2 > y1)
            if x2 > x1 and y2 > y1:
                box_color = (0, 255, 0) # Xanh lá cho trạng thái bình thường
                status_text = f"State: {label}" # Nhãn trạng thái gốc

                if is_falling_period:
                    box_color = (0, 0, 255) # Đỏ cho trạng thái ngã
                    if label == 8: # Giả sử nhãn 8 là giai đoạn đang ngã
                        status_text = "Falling (P1)"
                    elif label == 7: # Giả sử nhãn 7 là giai đoạn đã ngã
                        status_text = "Fallen (P2)"
                    else: # Các trường hợp khác trong giai đoạn ngã
                        status_text = "FALL DETECTED"
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
                # Đặt text trạng thái gần bounding box
                text_y_pos = y1 - 10 if y1 > 20 else y1 + (y2-y1) + 20 # Đảm bảo text không ra ngoài frame
                cv2.putText(processed_frame, status_text, (x1, text_y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
            # else: # Bounding box không hợp lệ, không vẽ gì
            #     print(f"Cảnh báo: Bounding box không hợp lệ tại frame {frame_num_counter}: x1={x1},y1={y1},x2={x2},y2={y2}")


        # Hiển thị thông tin chung trên frame
        cv2.putText(processed_frame, f"Frame: {frame_num_counter}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if is_falling_period:
             # Chỉ hiển thị "FALL PERIOD" nếu không có bbox cụ thể nào được vẽ cho giai đoạn ngã này
            if not (frame_num_counter in annotations and annotations[frame_num_counter]['x2'] > annotations[frame_num_counter]['x1']):
                cv2.putText(processed_frame, "FALL PERIOD (No BBox)", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
            
        out.write(processed_frame) # Ghi frame đã xử lý vào video output
        
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
    print(f"\nHoàn thành xử lý video! Đã xử lý {frame_num_counter} frames.")
    print(f"Video thành phẩm được lưu tại: '{output_video_path}'")
    print("--- HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ---")