import os

BENIGN_DIR = r"C:\Huy_Malware_AI_v2\data_raw\benign"
MAX_SIZE = 5242880 
MIN_SIZE = 10240   

def clean_benign_dataset():
    if not os.path.exists(BENIGN_DIR):
        print("Khong tim thay thu muc benign!")
        return

    files = [f for f in os.listdir(BENIGN_DIR) if os.path.isfile(os.path.join(BENIGN_DIR, f))]
    removed_count = 0
    kept_count = 0

    print("Dang loc du lieu Benign duoi 5MB...")
    
    for filename in files:
        filepath = os.path.join(BENIGN_DIR, filename)
        try:
            size = os.path.getsize(filepath)
            if size > MAX_SIZE or size < MIN_SIZE:
                os.remove(filepath)
                removed_count += 1
            else:
                kept_count += 1
        except Exception:
            pass

    print("\n" + "="*40)
    print(f"Da loai bo: {removed_count} file (Khong dat chuan 5MB)")
    print(f"Giu lai: {kept_count} file Benign chuẩn bài")
    print("="*40)

if __name__ == "__main__":
    clean_benign_dataset()