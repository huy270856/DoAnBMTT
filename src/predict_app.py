import os
import time
import datetime
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import filedialog, messagebox

# Khóa log ẩn thông báo từ TensorFlow ở Terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "malware_spectrogram_model.keras")
LOG_FILE_PATH = os.path.join(BASE_DIR, "msic_log.txt")
TEMP_IMG_PATH = os.path.join(BASE_DIR, "temp_predict.png")
CURRENT_VERSION = 2.9

# ================= 2. LOGIC AI (GIỮ NGUYÊN 100%) =================
class MalwarePredictor:
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {MODEL_PATH}")

    def create_spectrogram(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read(5242880)
            if len(raw_bytes) % 2 != 0: raw_bytes += b'\x00'
            
            signal = np.frombuffer(raw_bytes, dtype=np.int16)
            
            fig, ax = plt.subplots(figsize=(2.56, 2.56))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            fig.patch.set_facecolor('#000004')
            ax.set_facecolor('#000004')

            ax.specgram(signal, NFFT=2048, Fs=44100, 
                        window=np.hanning(2048), noverlap=1536, cmap='inferno')

            fig.savefig(TEMP_IMG_PATH, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            return True
        except Exception:
            return False

    def predict(self, file_path):
        if not self.create_spectrogram(file_path):
            return None, "Lỗi tạo ảnh"
            
        img = image.load_img(TEMP_IMG_PATH, target_size=(200, 200))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        if os.path.exists(TEMP_IMG_PATH): os.remove(TEMP_IMG_PATH)
        
        label = "MALWARE" if prediction > 0.5 else "AN TOÀN"
        conf = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        
        return {'label': label, 'confidence': conf, 'malware_prob': prediction * 100}, "OK"

# ================= 3. GIAO DIỆN SIÊU CẤP ĐẦY ĐỦ TÍNH NĂNG =================
class PredictApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("dark")
        self.title("MSIC AI Pro 2026")
        self.geometry("1050x760") # Nới rộng không gian để đặt Lá chắn đổi màu
        self.resizable(False, False)

        try:
            self.predictor = MalwarePredictor()
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Động", str(e))
            self.destroy()
            return

        self.is_scanning = False 

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_content()
        self.load_log_into_app()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#16191c")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        self.logo_lbl = ctk.CTkLabel(self.sidebar, text="🛡️ MSIC AI PRO", font=ctk.CTkFont(size=22, weight="bold", family="Arial"))
        self.logo_lbl.pack(pady=(40, 5), padx=20, anchor="w")
        
        self.sub_logo = ctk.CTkLabel(self.sidebar, text="Malware Detection System", text_color="#556475", font=("Arial", 12))
        self.sub_logo.pack(pady=(0, 40), padx=20, anchor="w")

        self.btn_scan = ctk.CTkButton(self.sidebar, text="🔍  Quét Hệ Thống", font=("Arial", 14, "bold"),
                                      fg_color="#1d6fdc", hover_color="#185cc5", height=42, corner_radius=8, command=self.browse_file)
        self.btn_scan.pack(fill="x", padx=20, pady=8)

        self.btn_clear_log = ctk.CTkButton(self.sidebar, text="🗑️  Xóa Nhật Ký", font=("Arial", 14),
                                          fg_color="transparent", text_color="#ff4d4d", hover_color="#2b1a1a", height=42, corner_radius=8, command=self.clear_log_file)
        self.btn_clear_log.pack(fill="x", padx=20, pady=8)

        self.info_lbl = ctk.CTkLabel(self.sidebar, text=f"Phiên bản: Ver {CURRENT_VERSION}\nTrạng thái: Sẵn sàng", 
                                     text_color="#556475", font=("Arial", 11), justify="left")
        self.info_lbl.pack(side="bottom", pady=25, padx=25, anchor="w")

    def setup_main_content(self):
        self.main_container = ctk.CTkFrame(self, corner_radius=12, fg_color="#0d0f11")
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # --- KHU VỰC BẢNG TRẠNG THÁI ---
        self.res_card = ctk.CTkFrame(self.main_container, fg_color="#16191c", height=380, corner_radius=12, border_width=1, border_color="#22272b")
        self.res_card.pack(pady=(20, 10), padx=30, fill="x")
        self.res_card.pack_propagate(False)

        # 🚀 TÍNH NĂNG MỚI: LÁ CHẮN BẢO MẬT ĐỔI MÀU TRỰC QUAN (Shield Icon) 
        self.shield_lbl = ctk.CTkLabel(self.res_card, text="🛡️", font=("Arial", 64))
        self.shield_lbl.pack(pady=(20, 5))

        self.status_title = ctk.CTkLabel(self.res_card, text="Chờ tải tệp tin phân tích...", font=("Arial", 18, "bold"), text_color="#a9b6c3")
        self.status_title.pack(pady=5)

        self.prob_bar = ctk.CTkProgressBar(self.res_card, width=500, height=10, corner_radius=5, fg_color="#22272b")
        self.prob_bar.pack(pady=10)
        self.prob_bar.set(0)

        self.detail_lbl = ctk.CTkLabel(self.res_card, text="Định dạng hỗ trợ: .exe, .dll", font=("Consolas", 13), text_color="#556475")
        self.detail_lbl.pack(pady=5)

        # --- KHUNG THÔNG TIN CHI TIẾT TỆP TIN ---
        self.meta_frame = ctk.CTkFrame(self.res_card, fg_color="#0d0f11", height=120, corner_radius=8)
        self.meta_frame.pack(padx=25, pady=10, fill="x")
        self.meta_frame.pack_propagate(False)
        
        self.meta_title = ctk.CTkLabel(self.meta_frame, text="📊 THÔNG SỐ CẤU TRÚC TỆP TIN", font=("Arial", 11, "bold"), text_color="#3B71B9")
        self.meta_title.place(x=15, y=10)
        
        self.lbl_size = ctk.CTkLabel(self.meta_frame, text="• Kích thước file : --", font=("Consolas", 12), text_color="#a9b6c3")
        self.lbl_size.place(x=15, y=35)
        
        self.lbl_type = ctk.CTkLabel(self.meta_frame, text="• Định dạng PE   : --", font=("Consolas", 12), text_color="#a9b6c3")
        self.lbl_type.place(x=15, y=62)
        
        self.lbl_stft = ctk.CTkLabel(self.meta_frame, text="• Xử lý tín hiệu : STFT (Hanning)", font=("Consolas", 12), text_color="#556475")
        self.lbl_stft.place(x=290, y=35)
        
        self.lbl_model = ctk.CTkLabel(self.meta_frame, text="• Phân loại AI   : MSIC CNN-3 Core", font=("Consolas", 12), text_color="#556475")
        self.lbl_model.place(x=290, y=62)

        # --- KHU VỰC NHẬT KÝ HỆ THỐNG ---
        self.log_title = ctk.CTkLabel(self.main_container, text="📋 Nhật ký lịch sử quét hệ thống", font=("Arial", 14, "bold"), text_color="#a9b6c3")
        self.log_title.pack(padx=30, anchor="w", pady=(5, 2))

        self.log_box = ctk.CTkTextbox(self.main_container, corner_radius=10, fg_color="#16191c", border_width=1, border_color="#22272b", font=("Consolas", 12), height=140)
        self.log_box.pack(padx=30, pady=5, fill="both", expand=True)
        self.log_box.configure(state="disabled")

        self.action_btn = ctk.CTkButton(self.main_container, text="Chọn File Quét...", width=160, height=38, font=("Arial", 13, "bold"),
                                        fg_color="#1d6fdc", hover_color="#185cc5", corner_radius=8, command=self.browse_file)
        self.action_btn.pack(pady=15)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Executable Files", "*.exe *.dll")])
        if path:
            self.status_title.configure(text=f"📂 Đang phân tích: {os.path.basename(path)}...", text_color="#e9af22")
            
            # Đổi lá chắn sang màu vàng nhấp nháy khi bắt đầu quét 
            self.shield_lbl.configure(text="⚙️", text_color="#e9af22", font=("Arial", 64))
            
            file_size_bytes = os.path.getsize(path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            self.lbl_size.configure(text=f"• Kích thước file : {file_size_mb:.2f} MB", text_color="#e9af22")
            
            ext = os.path.splitext(path)[1].upper()
            self.lbl_type.configure(text=f"• Định dạng PE   : Portable {ext} structure", text_color="#e9af22")
            
            self.is_scanning = True
            self.prob_bar.configure(progress_color="#e9af22")
            self.prob_bar.start()
            
            threading.Thread(target=self.radar_animation_loop, daemon=True).start()
            threading.Thread(target=self.run_ai, args=(path,), daemon=True).start()

    def radar_animation_loop(self):
        spin_icons = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]
        text_stages = [
            "Đang bóc tách cấu trúc PE nhị phân...          ",
            "Đang nạp ma trận tín hiệu byte...              ",
            "Đang dựng đồ thị Spectrogram màu...           ",
            "Đang áp dụng cửa sổ Hanning window...          ",
            "Đang nạp ảnh phổ tần vào mô hình CNN-3...      ",
            "Đang phân tích đặc trưng lớp tích chập...      ",
            "Đang phân loại ma trận mạng nơ-ron...          "
        ]
        idx = 0
        while self.is_scanning:
            icon = spin_icons[idx % len(spin_icons)]
            stage_text = text_stages[(idx // 4) % len(text_stages)]
            self.detail_lbl.configure(text=f"📡 [ {icon} ] {stage_text}", text_color="#e9af22")
            idx += 1
            time.sleep(0.1)

    def run_ai(self, path):
        result, status = self.predictor.predict(path)
        time.sleep(1.2)
        
        self.is_scanning = False 
        self.after(0, lambda: self.update_ui(result, path))

    def update_ui(self, result, path):
        self.prob_bar.stop()
        if not result:
            messagebox.showerror("Lỗi", "Không thể phân tích file này!")
            return

        safety_ratio = (100.0 - result['malware_prob']) / 100.0
        self.prob_bar.set(safety_ratio)
        
        if safety_ratio >= 0.75:
            color_hex = "#4dff88"
            shield_icon = "🛡️"     # Giữ nguyên khiên xanh lá cây khi sạch 
        elif safety_ratio >= 0.50:
            color_hex = "#99ff33"
            shield_icon = "🛡️"
        elif safety_ratio >= 0.25:
            color_hex = "#ff8533"
            shield_icon = "⚠️"     # Đổi sang chấm than cảnh báo mối đe dọa 
        else:
            color_hex = "#ff4d4d"
            shield_icon = "⚠️"     # Đỏ nguy hiểm 
            
        self.prob_bar.configure(progress_color=color_hex)
        self.shield_lbl.configure(text=shield_icon, text_color=color_hex) # Cập nhật Lá chắn động 
        
        self.lbl_size.configure(text_color="#a9b6c3")
        self.lbl_type.configure(text_color="#a9b6c3")
        
        if result['label'] == "MALWARE":
            self.status_title.configure(text="⚠️ PHÁT HIỆN MÃ ĐỘC ĐỘC HẠI!", text_color="#ff4d4d")
            self.detail_lbl.configure(text=f"💥 Mối đe dọa nguy hiểm! Mức độ AN TOÀN: {(safety_ratio * 100):.2f}%", text_color="#ff4d4d")
        else:
            self.status_title.configure(text="✅ FILE AN TOÀN TUYỆT ĐỐI", text_color="#4dff88")
            self.detail_lbl.configure(text=f"🛡️ Hệ thống an toàn! Mức độ AN TOÀN: {(safety_ratio * 100):.2f}%", text_color="#4dff88")
            
        now = time.strftime("%d/%m/%Y %H:%M:%S")
        log_entry = f"[{now}] [{result['label']}] {path}\n"
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)

        self.log_box.configure(state="normal")
        self.log_box.insert("1.0", log_entry)
        self.log_box.configure(state="disabled")

        if result['label'] == "MALWARE":
            confirm = messagebox.askyesno(
                "Cảnh Báo Mối Đe Dọa", 
                f"Phần mềm phát hiện nguy hiểm tại:\n{path}\n\nĐộ an toàn chỉ còn ở mức báo động: {(safety_ratio * 100):.2f}%\nBạn có muốn XÓA VĨNH VIỄN tệp tin này không?"
            )
            if confirm:
                try:
                    os.remove(path)
                    messagebox.showinfo("Thành Công", "🛡️ Đã tiêu diệt và loại bỏ hoàn toàn file mã độc khỏi hệ thống!")
                    
                    delete_log = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] [ĐÃ XÓA FILE MÃ ĐỘC] {path}\n"
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(delete_log)
                    
                    self.log_box.configure(state="normal")
                    self.log_box.insert("1.0", delete_log)
                    self.log_box.configure(state="disabled")
                    
                    self.status_title.configure(text="💥 MỐI ĐE DỌA ĐÃ ĐƯỢC XỬ LÝ", text_color="#4dff88")
                    self.detail_lbl.configure(text=f"File độc hại đã bị xóa bỏ hoàn toàn khỏi ổ cứng.", text_color="gray")
                    # Trả khiên về trạng thái an toàn sau khi đã diệt độc xong 
                    self.shield_lbl.configure(text="🛡️", text_color="#4dff88")
                except Exception as e:
                    messagebox.showerror("Lỗi Xóa File", f"Không thể xóa file: {e}")

    def load_log_into_app(self):
        if os.path.exists(LOG_FILE_PATH):
            try:
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                self.log_box.configure(state="normal")
                self.log_box.delete("1.0", "end")
                for line in reversed(lines):
                    self.log_box.insert("end", line)
                self.log_box.configure(state="disabled")
            except Exception:
                pass

    def clear_log_file(self):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa toàn bộ lịch sử nhật ký không?"):
            if os.path.exists(LOG_FILE_PATH):
                os.remove(LOG_FILE_PATH)
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.configure(state="disabled")
            messagebox.showinfo("Thông báo", "Đã xóa sạch nhật ký.")
            # Reset khiên về mặc định 
            self.shield_lbl.configure(text="🛡️", text_color="#a9b6c3")

if __name__ == "__main__":
    app = PredictApp()
    app.mainloop()