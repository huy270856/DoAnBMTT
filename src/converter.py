import numpy as np
from PIL import Image
import math
import os

def calculate_entropy_block(block):
    # SỬA LỖI TẠI ĐÂY: Kiểm tra mảng Numpy đúng cách
    if block is None or len(block) == 0: 
        return 0
    
    values, counts = np.unique(block, return_counts=True)
    probs = counts / len(block)
    entropy = -np.sum(probs * np.log2(probs + 1e-9)) # Thêm 1 số rất nhỏ để tránh lỗi log(0)
    return (entropy / 8.0) * 255

def file_to_rgb_image(file_path, save_path, img_size=256):
    try:
        target_len = img_size * img_size
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            if not raw_bytes: 
                return False
            data = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # Thuật toán ép dữ liệu cho file nhỏ
        if len(data) < target_len:
            repeats = (target_len // len(data)) + 1
            data = np.tile(data, repeats)[:target_len]
        else:
            data = data[:target_len]
        
        # Kênh Red: Dữ liệu thô
        red = data.reshape((img_size, img_size))
        
        # Kênh Green: Entropy
        green_flat = np.zeros(target_len)
        step = 16 
        for i in range(0, target_len, step):
            block = data[i:i+step]
            green_flat[i:i+step] = calculate_entropy_block(block)
        green = green_flat.reshape((img_size, img_size)).astype(np.uint8)
        
        # Kênh Blue: Vị trí
        blue = np.linspace(0, 255, target_len).reshape((img_size, img_size)).astype(np.uint8)
        
        # Gộp thành ảnh màu
        rgb_img = np.stack((red, green, blue), axis=-1)
        Image.fromarray(rgb_img, 'RGB').save(save_path)
        return True
    except Exception as e:
        print(f"❌ Lỗi tại file {os.path.basename(file_path)}: {e}")
        return False

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
input_base = os.path.join(root_dir, "data_raw")
output_base = os.path.join(root_dir, "images_dataset")

# Giữ nguyên tên folder của Huy nè
categories = ["begin", "malware"] 

# ... (Giữ nguyên phần trên của converter.py) ...

for cat in categories:
    in_dir = os.path.join(input_base, cat)
    out_dir = os.path.join(output_base, cat)
    
    if not os.path.exists(in_dir):
        continue
        
    os.makedirs(out_dir, exist_ok=True) 
    files = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    
    print(f"\n--- Processing: {cat.upper()} ---")
    
    skip_count = 0
    convert_count = 0

    for file_name in files:
        in_path = os.path.join(in_dir, file_name)
        out_name = file_name + ".png"
        out_path = os.path.join(out_dir, out_name)

        # 🕵️ KIỂM TRA: Nếu ảnh đã tồn tại thì bỏ qua luôn
        if os.path.exists(out_path):
            skip_count += 1
            continue 
            
        if file_to_rgb_image(in_path, out_path):
            print(f"✅ New: {file_name}")
            convert_count += 1

    print(f"📊 Kết quả: Mới ({convert_count}) - Đã có ({skip_count})")

print(f"\n--- XONG! Huy không cần đợi convert lại file cũ nữa nhé ---")