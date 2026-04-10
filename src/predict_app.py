import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Import lại hàm convert từ file cũ của Huy
from converter import file_to_rgb_image 

# 1. Cấu hình
IMG_SIZE = 256
MODEL_PATH = "models/malware_detector_model.h5"
# Tạo folder tạm để chứa ảnh khi test
TEMP_DIR = "temp_test"
os.makedirs(TEMP_DIR, exist_ok=True)

# 2. Tải mô hình đã train
if not os.path.exists(MODEL_PATH):
    print("❌ Không thấy file mô hình! Huy nhớ chạy train_model.py trước nhé.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["Benign (Lành tính)", "Malware (Mã độc)"]

def predict_file(file_path):
    print(f"\n🔍 Đang kiểm tra file: {os.path.basename(file_path)}")
    
    # Bước 1: Convert file exe sang ảnh png tạm thời
    temp_img_path = os.path.join(TEMP_DIR, "temp_predict.png")
    if not file_to_rgb_image(file_path, temp_img_path, IMG_SIZE):
        print("❌ Lỗi khi đọc file!")
        return

    # Bước 2: Nạp ảnh vào AI
    img = Image.open(temp_img_path).convert('RGB')
    img_array = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img_array = np.expand_dims(img_array, 0) # Thêm chiều batch
    img_array = img_array / 255.0 # Chuẩn hóa giống lúc train

    # Bước 3: Dự đoán
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"🤖 Kết quả: {result}")
    print(f"📊 Độ tin cậy: {confidence:.2f}%")

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Huy có thể kéo file exe cần test vào đây
    test_file = input("👉 Nhập đường dẫn file .exe cần test (hoặc kéo file vào đây): ").strip('"')
    if os.path.exists(test_file):
        predict_file(test_file)
    else:
        print("❌ File không tồn tại!")