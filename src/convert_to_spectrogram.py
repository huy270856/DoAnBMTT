import os
import numpy as np
import librosa
import librosa.display
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# ================= CẤU HÌNH =================
RAW_DIR = r"C:\Huy_Malware_AI_v2\data_raw"
IMG_DIR = r"C:\Huy_Malware_AI_v2\images_dataset"

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
TARGET_SIZE = (200, 200)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def file_to_spectrogram(src_path, dst_path):
    """Chuyển raw bytes → spectrogram màu (như Figure 3 paper)"""
    if os.path.exists(dst_path):
        return True
        
    try:
        # 1. Đọc raw bytes
        with open(src_path, 'rb') as f:
            raw_bytes = f.read()
            
        if len(raw_bytes) % 2 != 0:
            raw_bytes += b'\x00'
            
        # 2. Convert to time-domain signal
        signal = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        signal /= 32768.0
        
        # 3. STFT với Hanning window
        D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                         win_length=WIN_LENGTH, window='hann')
        
        # 4. Chuyển sang dB scale
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 5. Crop nếu cần
        if S_db.shape[0] > 500:
            S_db = S_db[:500, :]
        
        # 6. Tạo ảnh màu bằng matplotlib (đúng như paper)
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Dùng colormap 'inferno' hoặc 'viridis' như paper
        im = ax.imshow(S_db, aspect='auto', origin='lower', cmap='inferno')
        
        # 7. Lưu ảnh
        fig.savefig(dst_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logging.error(f"Lỗi {os.path.basename(src_path)}: {e}")
        return False

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    categories = ['benign', 'malware']
    
    for cat in categories:
        src_cat = os.path.join(RAW_DIR, cat)
        dst_cat = os.path.join(IMG_DIR, cat)
        os.makedirs(dst_cat, exist_ok=True)
        
        if not os.path.isdir(src_cat):
            continue
            
        files = [f for f in os.listdir(src_cat) if os.path.isfile(os.path.join(src_cat, f))]
        logging.info(f"🔄 Đang xử lý [{cat.upper()}]: {len(files)} files...")
        
        success, fail = 0, 0
        for fname in tqdm(files, desc=f"[{cat}]"):
            src = os.path.join(src_cat, fname)
            dst = os.path.join(dst_cat, f"{os.path.splitext(fname)[0]}.png")
            
            if file_to_spectrogram(src, dst):
                success += 1
            else:
                fail += 1
                
        logging.info(f"✅ [{cat}] Hoàn tất: Thành công {success} | Lỗi {fail}")

if __name__ == "__main__":
    main()