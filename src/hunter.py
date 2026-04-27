import requests
import os
import zipfile
import io
import time
import pyzipper
import matplotlib.pyplot as plt

API_KEY = "5f4fb7dada95e41d569bb58592c9f485260b72f01ce78eac"
SAVE_DIR = r"C:\Huy_Malware_AI_v2\data_raw\malware"
TARGET_TOTAL = 8000
LIMIT_PER_QUERY = 1000

os.makedirs(SAVE_DIR, exist_ok=True)

def download_samples(query_data, description, needed):
    if needed <= 0:
        return 0
        
    url = "https://mb-api.abuse.ch/api/v1/"
    headers = {'Auth-Key': API_KEY.strip()}
    
    print(f"\n📡 {description}...")
    try:
        res = requests.post(url, data=query_data, headers=headers, timeout=60)
        
        if res.status_code != 200:
            print(f"⚠️ HTTP {res.status_code}")
            return 0
        
        json_response = res.json()
        if json_response.get('query_status') != 'ok':
            print(f"⚠️ API error: {json_response.get('query_status')}")
            return 0
        
        samples = json_response.get('data', [])
        if not samples:
            print("❌ Không tìm thấy mẫu nào")
            return 0
            
        samples_to_download = samples[:needed]
        print(f"🔎 Tìm thấy {len(samples)} mẫu. Bắt đầu tải {len(samples_to_download)} mẫu...")
        
        success = 0
        for s in samples_to_download:
            sha256 = s['sha256_hash']
            file_name = s.get('file_name', sha256)
            
            file_path = os.path.join(SAVE_DIR, file_name)
            if os.path.exists(file_path):
                print(f"⏩ Hàng đã có trong kho, tự động bỏ qua: {file_name}")
                continue
            
            try:
                f_res = requests.post(
                    url, 
                    data={'query': 'get_file', 'sha256_hash': sha256}, 
                    headers=headers, 
                    timeout=60
                )
                
                if f_res.status_code == 200 and f_res.content.startswith(b'PK'):
                    try:
                        with pyzipper.AESZipFile(io.BytesIO(f_res.content)) as z:
                            z.pwd = b'infected'
                            z.extractall(path=SAVE_DIR)
                        success += 1
                        print(f"✅ [{success}/{len(samples_to_download)}] Đã bế lên xe: {file_name}")
                        time.sleep(1)
                    except Exception as zip_err:
                        print(f"⚠️ Lỗi giải nén file {sha256[:10]}: {zip_err}")
                else:
                    print(f"⚠️ Failed hoặc không phải file ZIP: {sha256[:16]}")
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
                
        return success
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return 0

if __name__ == "__main__":
    total_downloaded = 0
    
    families = [
        "Emotet", "AgentTesla", "Remcos", "Formbook", 
        "CobaltStrike", "QakBot", "TrickBot", "GuLoader", 
        "IcedID", "RedLineStealer", "Loki", "NanoCore"
    ]
    
    for family in families:
        if total_downloaded >= TARGET_TOTAL:
            break
            
        needed = TARGET_TOTAL - total_downloaded
        total_downloaded += download_samples(
            {'query': 'get_siginfo', 'signature': family, 'limit': LIMIT_PER_QUERY},
            f"📥 Đang truy nã băng đảng: {family}",
            needed
        )
        time.sleep(2)
    
    if total_downloaded < TARGET_TOTAL:
        needed = TARGET_TOTAL - total_downloaded
        total_downloaded += download_samples(
            {'query': 'get_file_type', 'file_type': 'exe', 'limit': LIMIT_PER_QUERY},
            "📥 Đang vét thêm file EXE cho đủ KPI",
            needed
        )
    
    print(f"\n✨ HOÀN THÀNH XUẤT SẮC! Đã tóm gọn {total_downloaded} mã độc mới vào: {SAVE_DIR}")