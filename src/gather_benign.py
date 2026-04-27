import os
import shutil

TARGET_COUNT = 6000
DEST_DIR = r"C:\Huy_Malware_AI_v2\data_raw\benign"
MAX_FILE_SIZE = 4194304 
MIN_FILE_SIZE = 10240

local_app_data = os.environ.get('LOCALAPPDATA', '')
roaming_app_data = os.environ.get('APPDATA', '')
prog_files = os.environ.get('PROGRAMFILES', r'C:\Program Files')
prog_files_x86 = os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')

SOURCE_DIRS = [
    local_app_data,
    roaming_app_data,
    prog_files,
    prog_files_x86,
    r"C:\Windows\System32\drivers",
    r"C:\Windows\SysWOW64"
]

def gather_diverse_benign():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    existing_files = len([f for f in os.listdir(DEST_DIR) if os.path.isfile(os.path.join(DEST_DIR, f))])
    needed = TARGET_COUNT - existing_files
    
    if needed <= 0:
        print(f"Kho benign da du {existing_files} file.")
        return

    print(f"Can thu thap them {needed} file. Bat dau quet...")
    
    count = 0
    for src_dir in SOURCE_DIRS:
        if not os.path.exists(src_dir):
            continue
            
        print(f"Dang quet: {src_dir}")
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.exe', '.dll')):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(DEST_DIR, file)
                    
                    if not os.path.exists(dst_path):
                        try:
                            size = os.path.getsize(src_path)
                            if MIN_FILE_SIZE <= size <= MAX_FILE_SIZE:
                                shutil.copy2(src_path, dst_path)
                                count += 1
                                
                                if count % 100 == 0:
                                    print(f"Da copy: {count}/{needed} files")
                                
                                if count >= needed:
                                    print(f"\nHoan thanh target {TARGET_COUNT} file.")
                                    return
                        except Exception:
                            pass

if __name__ == "__main__":
    gather_diverse_benign()