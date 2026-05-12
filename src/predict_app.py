import os
import time
import datetime
import threading
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# --- CẤU HÌNH GIAO DIỆN MẶC ĐỊNH ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# --- THÔNG SỐ CẬP NHẬT ---
CURRENT_VERSION = 2.2
UPDATE_URL = "https://raw.githubusercontent.com/huy270856/MSIC_Update_Server/refs/heads/main/version.json" # <--- DÁN LINK RAW VÀO ĐÂY

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_PATH = r"C:\Huy_Malware_AI_v2\models\malware_spectrogram_model.keras"
TEMP_IMG_PATH = r"C:\Huy_Malware_AI_v2\temp_predict.png"
LOG_FILE_PATH = r"C:\Huy_Malware_AI_v2\msic_log.txt"

# --- THÔNG SỐ STFT ---
MAX_SIZE = 5242880
SAMPLING_RATE = 44100
NFFT_WINDOW = 2048
HOP_LENGTH = 512
NOVERLAP = NFFT_WINDOW - HOP_LENGTH
TARGET_SIZE = (200, 200)

class MalwarePredictor:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Không tìm thấy model tại {MODEL_PATH}")
            
    def create_temp_spectrogram(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read(MAX_SIZE)
            if len(raw_bytes) % 2 != 0:
                raw_bytes += b'\x00'
            signal = np.frombuffer(raw_bytes, dtype=np.int16)
            fig, ax = plt.subplots(figsize=(2.56, 2.56))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            fig.patch.set_facecolor('#000004')
            ax.set_facecolor('#000004')
            ax.specgram(signal, NFFT=NFFT_WINDOW, Fs=SAMPLING_RATE, window=np.hanning(NFFT_WINDOW), noverlap=NOVERLAP, cmap='inferno')
            fig.savefig(TEMP_IMG_PATH, dpi=100, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
            plt.close(fig)
            return True
        except Exception:
            return False

    def predict(self, file_path):
        if not os.path.exists(file_path):
            return None, "File không tồn tại"
        success = self.create_temp_spectrogram(file_path)
        if not success:
            return None, "Không thể tạo spectrogram từ file này"
        try:
            img = image.load_img(TEMP_IMG_PATH, target_size=TARGET_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            if os.path.exists(TEMP_IMG_PATH):
                os.remove(TEMP_IMG_PATH)
            if prediction > 0.5:
                return {'label': "MALWARE", 'confidence': prediction * 100, 'malware_prob': prediction * 100, 'benign_prob': (1 - prediction) * 100}, "OK"
            else:
                return {'label': "BENIGN", 'confidence': (1 - prediction) * 100, 'malware_prob': prediction * 100, 'benign_prob': (1 - prediction) * 100}, "OK"
        except Exception as e:
            return None, str(e)

# === GIAO DIỆN MSIC ===
class PredictApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MSIC AI Pro 2026")
        self.geometry("750x520")
        self.resizable(False, False)
        try:
            self.predictor = MalwarePredictor()
        except Exception as e:
            messagebox.showerror("Lỗi Load Model", str(e))
            self.destroy()
            return
        self.setup_ui()

    def setup_ui(self):
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#1E539A", height=60)
        self.header_frame.pack(fill="x", side="top")
        self.header_frame.pack_propagate(False)
        self.logo_label = ctk.CTkLabel(self.header_frame, text="🛡️ MSIC", font=ctk.CTkFont(family="Arial", size=20, weight="bold"), text_color="white")
        self.logo_label.pack(side="left", padx=15, pady=10)
        self.sub_logo = ctk.CTkLabel(self.header_frame, text=f"AI Malware Detection | Ver {CURRENT_VERSION}", font=ctk.CTkFont(family="Arial", size=12), text_color="white")
        self.sub_logo.pack(side="left", pady=10)

        self.tabview = ctk.CTkTabview(self, width=700, corner_radius=8, command=self.on_tab_change)
        self.tabview.pack(padx=15, pady=(10, 5), fill="both", expand=True)
        self.tabview.add("Tùy chọn Quét")
        self.tabview.add("Nhật ký Hệ thống")

        # --- TAB 1 ---
        tab_scan = self.tabview.tab("Tùy chọn Quét")
        self.file_frame = ctk.CTkFrame(tab_scan, fg_color="transparent")
        self.file_frame.pack(fill="x", padx=5, pady=5)
        self.file_path_var = ctk.StringVar()
        self.file_entry = ctk.CTkEntry(self.file_frame, textvariable=self.file_path_var, placeholder_text="Đường dẫn file cần quét...", width=450, height=35)
        self.file_entry.pack(side="left", padx=(0, 10))
        self.browse_btn_top = ctk.CTkButton(self.file_frame, text="Mở File...", command=self.browse_file, width=100, height=35, fg_color="#6C757D")
        self.browse_btn_top.pack(side="left")

        self.result_frame = ctk.CTkFrame(tab_scan, corner_radius=8)
        self.result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.result_text = ctk.CTkTextbox(self.result_frame, font=("Courier New", 13), wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.progress = ctk.CTkProgressBar(tab_scan, orientation="horizontal", mode="indeterminate", height=10)
        self.progress.pack(fill="x", padx=5, pady=5)
        self.progress.set(0)

        # --- TAB 2: SCROLLABLE FRAME ---
        tab_log = self.tabview.tab("Nhật ký Hệ thống")
        tab_log.grid_columnconfigure(0, weight=1)
        tab_log.grid_columnconfigure(1, weight=1)
        tab_log.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(tab_log, text="📁 Lịch sử quét", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")
        ctk.CTkLabel(tab_log, text="⬇️ Tải về (24h)", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=1, padx=5, pady=(5, 0), sticky="w")
        
        self.log_scroll = ctk.CTkScrollableFrame(tab_log, fg_color="#1E1E1E", corner_radius=6)
        self.log_scroll.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.download_scroll = ctk.CTkScrollableFrame(tab_log, fg_color="#F1F3F5", corner_radius=6)
        self.download_scroll.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # 3. BOTTOM BAR 
        self.footer_frame = ctk.CTkFrame(self, corner_radius=0, height=70, fg_color="transparent")
        self.footer_frame.pack(fill="x", side="bottom", padx=15, pady=(0, 15))
        self.version_lbl = ctk.CTkLabel(self.footer_frame, text=f"Ver {CURRENT_VERSION} - Sẵn sàng", text_color="gray", font=("Arial", 11))
        self.version_lbl.pack(side="left")

        self.exit_btn = ctk.CTkButton(self.footer_frame, text="❌ Thoát", command=self.destroy, fg_color="#3B71B9", width=90, height=45, font=("Arial", 14, "bold"), corner_radius=2)
        self.exit_btn.pack(side="right", padx=3)
        
        self.scan_btn = ctk.CTkButton(self.footer_frame, text="🔍 Quét", command=self.start_prediction, fg_color="#3B71B9", width=90, height=45, font=("Arial", 14, "bold"), corner_radius=2)
        self.scan_btn.pack(side="right", padx=3)
        
        self.update_btn = ctk.CTkButton(self.footer_frame, text="🔄 Kiểm tra cập nhật", command=self.check_update, fg_color="#3B71B9", width=160, height=45, font=("Arial", 14, "bold"), corner_radius=2)
        self.update_btn.pack(side="right", padx=3)

        self.help_btn = ctk.CTkButton(self.footer_frame, text="🙋 Trợ giúp", command=self.show_help, fg_color="#3B71B9", width=100, height=45, font=("Arial", 14, "bold"), corner_radius=2)
        self.help_btn.pack(side="right", padx=3)

    # --- HÀM XỬ LÝ NHẬT KÝ ---
    def save_to_log(self, file_path, status_label):
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        log_entry = f"[{now}] [{status_label}] {file_path}\n"
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except: pass

    def load_log_data(self):
        for widget in self.log_scroll.winfo_children():
            widget.destroy()
            
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                logs = f.readlines()
                for line in reversed(logs):
                    line = line.strip()
                    if not line: continue
                    
                    row = ctk.CTkFrame(self.log_scroll, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    
                    text_color = "#E74C3C" if "[MÃ ĐỘC" in line else "#2ECC71"
                    
                    try:
                        parts = line.split("] ", 2)
                        if len(parts) == 3:
                            time_str = parts[0] + "]"
                            status_str = parts[1] + "]"
                            fpath = parts[2]
                            fname = os.path.basename(fpath)
                            
                            # Thu gọn tên file nếu quá dài
                            if len(fname) > 28:
                                fname = fname[:12] + "..." + fname[-12:]
                                
                            display_text = f"{time_str} {status_str} {fname}"
                            
                            # VẼ NÚT BẤM TRƯỚC ĐỂ CHIẾM CHỖ BÊN PHẢI
                            if "[MÃ ĐỘC" in line:
                                del_btn = ctk.CTkButton(row, text="🗑️ Xóa", width=50, height=24, fg_color="#E74C3C", hover_color="#C0392B", font=("Arial", 10, "bold"), command=lambda p=fpath, r=row: self.delete_malware_file(p, r))
                                del_btn.pack(side="right", padx=5)
                            
                            # VẼ LABEL SAU ĐỂ NÓ LẤY PHẦN TRỐNG CÒN LẠI
                            lbl = ctk.CTkLabel(row, text=display_text, text_color=text_color, font=("Consolas", 11), anchor="w", justify="left")
                            lbl.pack(side="left", fill="x", expand=True)
                    except: pass
        else:
            ctk.CTkLabel(self.log_scroll, text="Chưa có lịch sử quét.", text_color="white", font=("Consolas", 11)).pack()

    def delete_malware_file(self, file_path, row_widget):
        if messagebox.askyesno("Xóa file", f"Bạn có chắc chắn muốn xóa tận gốc file này khỏi máy tính không?\n\nĐường dẫn: {file_path}"):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    messagebox.showinfo("Thành công", "Đã tiêu diệt file mã độc thành công!")
                    row_widget.destroy() 
                else:
                    messagebox.showwarning("Lỗi", "File không còn tồn tại trên máy (có thể đã bị xóa trước đó).")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể xóa file. File có thể đang được chạy ngầm.\nChi tiết: {e}")

    def load_download_history(self):
        for widget in self.download_scroll.winfo_children():
            widget.destroy()
            
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        now = time.time()
        recent_files = []
        try:
            if os.path.exists(downloads_path):
                for f in os.listdir(downloads_path):
                    filepath = os.path.join(downloads_path, f)
                    if os.path.isfile(filepath) and (now - os.path.getmtime(filepath) <= 86400):
                        recent_files.append((f, filepath, os.path.getmtime(filepath)))
                recent_files.sort(key=lambda x: x[2], reverse=True)
                
                if not recent_files:
                    ctk.CTkLabel(self.download_scroll, text="Không có tệp tải về trong 24h.", font=("Consolas", 11), text_color="black").pack()
                    return
                
                for fname, fpath, mtime in recent_files:
                    row = ctk.CTkFrame(self.download_scroll, fg_color="transparent")
                    row.pack(fill="x", pady=2)
                    
                    # Thu gọn tên file tải về nếu quá dài
                    if len(fname) > 28:
                        display_fname = fname[:12] + "..." + fname[-12:]
                    else:
                        display_fname = fname
                    
                    time_str = datetime.datetime.fromtimestamp(mtime).strftime('%H:%M')
                    
                    # VẼ NÚT TRƯỚC
                    scan_btn = ctk.CTkButton(row, text="🔍 Quét nhanh", width=80, height=24, fg_color="#3B71B9", hover_color="#1E539A", font=("Arial", 10, "bold"), command=lambda p=fpath: self.quick_scan(p))
                    scan_btn.pack(side="right", padx=5)
                    
                    # VẼ LABEL SAU
                    lbl = ctk.CTkLabel(row, text=f"[{time_str}] {display_fname}", font=("Consolas", 11), text_color="black", anchor="w")
                    lbl.pack(side="left", fill="x", expand=True)
            else:
                ctk.CTkLabel(self.download_scroll, text="Không tìm thấy thư mục Downloads.", font=("Consolas", 11), text_color="black").pack()
        except Exception as e:
            ctk.CTkLabel(self.download_scroll, text=f"Lỗi: {str(e)}", font=("Consolas", 11), text_color="black").pack()

    def quick_scan(self, file_path):
        self.tabview.set("Tùy chọn Quét")
        self.file_path_var.set(file_path)
        self.start_prediction()

    # --- CÁC HÀM CẬP NHẬT & QUÉT ---
    def check_update(self):
        self.update_btn.configure(state="disabled", text="Đang kiểm tra...")
        self.result_text.insert("end", "[HỆ THỐNG] Đang kết nối tới máy chủ để kiểm tra phiên bản...\n")
        self.progress.start()
        threading.Thread(target=self.real_update_check, daemon=True).start()

    def real_update_check(self):
        try:
            time.sleep(1.5) 
            response = requests.get(UPDATE_URL, timeout=5)
            if response.status_code == 200:
                data = response.json()
                latest_version = data.get("version", 0.0)
                download_url = data.get("model_url", "")
                release_notes = data.get("message", "Bản cập nhật mới.")

                if float(latest_version) > CURRENT_VERSION:
                    self.after(0, self.prompt_update, latest_version, release_notes, download_url)
                else:
                    self.after(0, self.show_up_to_date)
            else:
                self.after(0, lambda: self.result_text.insert("end", "[LỖI] Server từ chối kết nối. Hãy kiểm tra lại đường link JSON.\n\n"))
                self.after(0, self.reset_update_btn)
        except Exception as e:
            self.after(0, lambda: self.result_text.insert("end", f"[LỖI MẠNG] Không thể kết nối. Chi tiết: {str(e)}\n\n"))
            self.after(0, self.reset_update_btn)

    def prompt_update(self, latest_version, release_notes, download_url):
        self.reset_update_btn()
        self.result_text.insert("end", f"[CẬP NHẬT] Tìm thấy phiên bản {latest_version}!\nChi tiết: {release_notes}\n\n", "benign_tag")
        answer = messagebox.askyesno("Có bản cập nhật mới!", f"Phát hiện bản cập nhật: Version {latest_version}\n\nNội dung: {release_notes}\n\nBạn có muốn tải AI Model mới về không?")
        if answer:
            self.result_text.insert("end", f"[HỆ THỐNG] Đang chuẩn bị tải file từ: {download_url}...\n\n")

    def show_up_to_date(self):
        self.reset_update_btn()
        self.result_text.insert("end", "[HỆ THỐNG] Đã kiểm tra xong. Hiện tại chưa có bản cập nhật mới.\n\n")
        messagebox.showinfo("Kiểm tra phiên bản", "Hiện tại chưa có bản cập nhật mới!\n\nBạn đang sử dụng phiên bản MSIC mới nhất.")

    def reset_update_btn(self):
        self.progress.stop()
        self.update_btn.configure(state="normal", text="🔄 Kiểm tra cập nhật")

    def on_tab_change(self):
        current_tab = self.tabview.get()
        if current_tab == "Nhật ký Hệ thống":
            self.footer_frame.pack_forget()
            self.load_log_data()
            self.load_download_history()
        else:
            self.footer_frame.pack(fill="x", side="bottom", padx=15, pady=(0, 15))

    def show_help(self):
        info_text = "MALWARE SPECTROGRAM CLASSIFIER (MSIC)\n\nTÁC GIẢ: Nguyễn Lê Nhật Huy\nSĐT: 0858340159\nEmail: nguyenlenhathuy2021@gmail.com"
        messagebox.showinfo("Trợ giúp & Liên hệ", info_text)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Executables", "*.exe *.dll *.bat"), ("All files", "*.*")])
        if file_path:
            self.file_path_var.set(file_path)
            self.result_text.delete("1.0", "end")
            self.result_text.insert("end", f"Đã sẵn sàng quét: {os.path.basename(file_path)}")
    
    def start_prediction(self):
        file_path = self.file_path_var.get().strip('"')
        if not file_path or not os.path.exists(file_path): return
        self.scan_btn.configure(state="disabled")
        self.progress.start()
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "[*] Đang trích xuất Spectrogram...\n")
        threading.Thread(target=self.run_prediction, args=(file_path,), daemon=True).start()
    
    def run_prediction(self, file_path):
        try:
            result, status = self.predictor.predict(file_path)
            if status == "OK":
                self.after(0, self.display_result, file_path, result)
        finally:
            self.after(0, self.prediction_complete)
            
    def display_result(self, file_path, result):
        self.result_text.insert("end", "-"*50 + "\n\n")
        self.result_text.tag_config("malware_tag", foreground="#E74C3C")
        self.result_text.tag_config("benign_tag", foreground="#27AE60")
        
        if result['label'] == "MALWARE":
            self.result_text.insert("end", f"⚠️ PHÁT HIỆN MÃ ĐỘC !!!\n", "malware_tag")
            self.save_to_log(file_path, "MÃ ĐỘC ") 
            if messagebox.askyesno("Cảnh báo", "Phát hiện mã độc! Xoá file ngay?"):
                try: os.remove(file_path)
                except: pass
        else:
            self.result_text.insert("end", f"✅ FILE AN TOÀN\n", "benign_tag")
            self.save_to_log(file_path, "AN TOÀN") 
            
        self.result_text.insert("end", f"\n[AI Phân Tích]\n- Malware: {result['malware_prob']:.2f}%\n- An toàn: {result['benign_prob']:.2f}%")

    def prediction_complete(self):
        self.progress.stop()
        self.scan_btn.configure(state="normal")

if __name__ == "__main__":
    app = PredictApp()
    app.mainloop()