import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
DATA_DIR = r"C:\Huy_Malware_AI_v2\images_dataset"
MODEL_DIR = r"C:\Huy_Malware_AI_v2\models"
MODEL_PATH = os.path.join(MODEL_DIR, "msic_cnn3_binary.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= GPU CONFIG =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print(f"⚠️  GPU Error: {e}")

# ================= HYPERPARAMETERS =================
IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

# ================= DATA GENERATORS =================
print("\n📊 Loading datasets...")

# Train: augmentation + validation_split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% validation
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Test: chỉ rescale (dùng 20% còn lại từ validation_split)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False
)

print(f"\n📈 Dataset Info:")
print(f"   Train samples: {train_generator.samples}")
print(f"   Val samples:   {val_generator.samples}")
print(f"   Classes: {train_generator.class_indices}")

# ================= MODEL CNN_3 =================
print("\n🏗️  Building CNN_3 model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Summary:")
model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# ================= TRAINING =================
print("\n🚀 Bắt đầu training...")
print("="*60)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ================= ĐÁNH GIÁ =================
print("\n" + "="*60)
print("📊 KẾT QUẢ TRAINING")
print("="*60)

# Lấy kết quả cuối cùng
final_train_loss, final_train_acc = model.evaluate(train_generator, verbose=0)
final_val_loss, final_val_acc = model.evaluate(val_generator, verbose=0)

print(f"\n✅ Train Accuracy: {final_train_acc*100:.2f}%")
print(f"   Train Loss:     {final_train_loss:.4f}")
print(f"\n✅ Val Accuracy:   {final_val_acc*100:.2f}%")
print(f"   Val Loss:       {final_val_loss:.4f}")

# ================= VẼ BIỂU ĐỒ =================
print("\n📊 Vẽ biểu đồ training history...")

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=300)
plt.show()

# ================= KẾT QUẢ CUỐI CÙNG =================
print("\n" + "="*60)
print("✅ HOÀN TẤT!")
print("="*60)
print(f"\n📁 Model lưu tại: {MODEL_PATH}")
print(f"📈 Training History: {os.path.join(MODEL_DIR, 'training_history.png')}")
print(f"\n🎯 Kết quả cuối cùng:")
print(f"   Train Accuracy: {final_train_acc*100:.2f}%")
print(f"   Val Accuracy:   {final_val_acc*100:.2f}%")
print("\n💡 So với paper MSIC:")
print(f"   Paper (binary): 96% accuracy")
print(f"   Model của bạn:  {final_val_acc*100:.2f}% validation accuracy")