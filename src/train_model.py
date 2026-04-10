import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# 1. CẤU HÌNH ĐƯỜNG DẪN
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATASET_PATH = os.path.join(project_root, "images_dataset")
IMG_SIZE = 256

def load_custom_dataset(path, img_size):
    images = []
    labels = []
    # Chị để cả 2 tên folder cho chắc, máy em có cái nào nó lấy cái đó
    possible_benign = ["benign", "begin"]
    class_names = ["benign", "malware"]
    
    # Nạp ảnh lành tính
    found_benign = False
    for b_name in possible_benign:
        b_path = os.path.join(path, b_name)
        if os.path.exists(b_path):
            for f in os.listdir(b_path):
                if f.endswith(".png"):
                    img = Image.open(os.path.join(b_path, f)).convert('RGB')
                    images.append(np.array(img.resize((img_size, img_size))))
                    labels.append(0)
                    found_benign = True
            if found_benign: break

    # Nạp ảnh mã độc
    m_path = os.path.join(path, "malware")
    if os.path.exists(m_path):
        for f in os.listdir(m_path):
            if f.endswith(".png"):
                img = Image.open(os.path.join(m_path, f)).convert('RGB')
                images.append(np.array(img.resize((img_size, img_size))))
                labels.append(1)

    return np.array(images), np.array(labels), class_names

# 2. NẠP DỮ LIỆU
print(f"📂 Đang nạp ảnh từ: {DATASET_PATH}")
X, y, class_names = load_custom_dataset(DATASET_PATH, IMG_SIZE)

if len(X) == 0:
    print("❌ Vẫn không thấy ảnh PNG nào! Huy kiểm tra folder images_dataset nhé.")
    exit()

print(f"✅ Đã nạp thành công {len(X)} ảnh!")

# Chia dữ liệu: Nếu ảnh ít quá thì test_size nhỏ thôi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=123)

# 3. XÂY DỰNG MÔ HÌNH CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(2) 
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. HUẤN LUYỆN
print("\n🚀 AI BẮT ĐẦU HỌC BÀI NÈ HUY...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
# Vẽ biểu đồ để xem AI học hành ra sao
plt.figure(figsize=(12, 4))

# Biểu đồ độ chính xác (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Độ chính xác qua các vòng học')
plt.legend()

# Biểu đồ lỗi (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Độ lỗi qua các vòng học')
plt.legend()

plt.show()

# 5. LƯU MÔ HÌNH
os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
save_path = os.path.join(project_root, "models", "malware_detector_model.h5")
model.save(save_path)

print(f"\n✅ THÀNH CÔNG! Bộ não đã lưu tại: {save_path}")