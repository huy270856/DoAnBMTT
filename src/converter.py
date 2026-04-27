import os
import numpy as np
import matplotlib.pyplot as plt
import logging

RAW_DIR = r"C:\Huy_Malware_AI_v2\data_raw"
IMG_DIR = r"C:\Huy_Malware_AI_v2\images_dataset"
LOG_FILE = r"C:\Huy_Malware_AI_v2\convert_log.txt"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')

MAX_SIZE = 5242880
SAMPLING_RATE = 44100
NFFT_WINDOW = 2048
HOP_LENGTH = 512
NOVERLAP = NFFT_WINDOW - HOP_LENGTH

def convert_to_spectrogram(src_path, dst_path):
    if os.path.exists(dst_path):
        return "SKIPPED_EXISTS"

    try:
        file_size = os.path.getsize(src_path)
        
        if file_size < 10240:
            logging.info(f"[SKIPPED_TOO_SMALL] {src_path} ({file_size} bytes)")
            return "SKIPPED_SMALL"

        with open(src_path, 'rb') as f:
            raw_bytes = f.read(MAX_SIZE)

        if len(raw_bytes) % 2 != 0:
            raw_bytes += b'\x00'

        signal = np.frombuffer(raw_bytes, dtype=np.int16)

        fig, ax = plt.subplots(figsize=(2.56, 2.56))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')

        fig.patch.set_facecolor('#000004')
        ax.set_facecolor('#000004')

        ax.specgram(signal, NFFT=NFFT_WINDOW, Fs=SAMPLING_RATE, 
                    window=np.hanning(NFFT_WINDOW), noverlap=NOVERLAP, cmap='inferno')

        fig.savefig(dst_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
        plt.close(fig)

        logging.info(f"[SUCCESS] {src_path}")
        return "SUCCESS"

    except Exception as e:
        logging.error(f"[ERROR] {src_path} - {str(e)}")
        return "ERROR"

def main():
    print(f"Bat dau convert, check log chi tiet tai: {LOG_FILE}")
    for category in ['benign', 'malware']:
        cat_raw_dir = os.path.join(RAW_DIR, category)
        cat_img_dir = os.path.join(IMG_DIR, category)
        os.makedirs(cat_img_dir, exist_ok=True)

        if not os.path.exists(cat_raw_dir):
            continue

        files = [f for f in os.listdir(cat_raw_dir) if os.path.isfile(os.path.join(cat_raw_dir, f))]
        total = len(files)
        
        for i, filename in enumerate(files):
            src_path = os.path.join(cat_raw_dir, filename)
            dst_path = os.path.join(cat_img_dir, f"{filename}.png")
            convert_to_spectrogram(src_path, dst_path)
            
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"Da xu ly {i + 1}/{total} files ({category})")

if __name__ == "__main__":
    main()