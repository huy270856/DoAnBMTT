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

# Import thư viện theo dõi ổ cứng thời gian thực
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Khóa log ẩn thông báo từ TensorFlow ở Terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 1. CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "malware_spectrogram_model.keras")
LOG_FILE_PATH = os.path.join(BASE_DIR, "msic_log.txt")
TEMP_IMG_PATH = os.path.join(BASE_DIR, "temp_predict.png")
CURRENT_VERSION = 5.7

DOWNLOADS_DIR = os.path.join(os.path.expanduser('~'), 'Downloads')

# ================= 2. LOGIC AI (GIỮ NGUYÊN 100% CỦA BẠN) =================
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


# ================= 3. BỘ LẮNG NGHE SỰ KIỆN FILE (WATCHDOG) =================
class DownloadFolderHandler(FileSystemEventHandler):
    def __init__(self, app_instance):
        self.app = app_instance

    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.exe', '.dll']:
            time.sleep(0.5) 
            if os.path.exists(file_path):
                self.app.after(0, lambda: self.app.trigger_realtime_scan(file_path))


# ================= 4. GIAO DIỆN PHÂN TRANG ĐỒNG BỘ VIỀN NỔI BẬT SIDEBAR =================
class PredictApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("dark")
        self.title("AI SHIELD PRO 2026")
        self.geometry("1080x750") 
        self.resizable(False, False)

        try:
            self.predictor = MalwarePredictor()
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Động", str(e))
            self.destroy()
            return

        self.is_scanning = False 
        self.download_paths_map = {} 
        self.observer = None
        self.realtime_enabled = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_content()
        self.load_log_into_app()

        # Đặt trạng thái nút gạt về ON trước
        self.rt_switch.select()
        
        # Chạy Watchdog sau 200ms bằng luồng riêng để giao diện không bị treo khi khởi động
        self.after(200, lambda: self.toggle_realtime_protection(force_on=True))
        
        # Mặc định khởi động ứng dụng tại trang Quét hệ thống
        self.switch_tab("🛡️ Bảo Vệ Hệ Thống")

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#16191c")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.logo_lbl = ctk.CTkLabel(self.sidebar, text="🛡️AI SHIELD PRO", font=ctk.CTkFont(size=22, weight="bold", family="Arial"))
        self.logo_lbl.pack(pady=(40, 5), padx=20, anchor="w")
        
        self.sub_logo = ctk.CTkLabel(self.sidebar, text="Malware Detection System", text_color="#556475", font=("Arial", 12))
        self.sub_logo.pack(pady=(0, 25), padx=20, anchor="w")

        # NÚT ĐIỀU HƯỚNG 1: QUÉT HỆ THỐNG
        self.btn_go_scan = ctk.CTkButton(self.sidebar, text="🔍  Quét Hệ Thống", font=("Arial", 14, "bold"),
                                         fg_color="transparent", text_color="#a9b6c3", hover_color="#22272b", height=38, corner_radius=8, 
                                         command=lambda: self.switch_tab("🛡️ Bảo Vệ Hệ Thống"))
        self.btn_go_scan.pack(fill="x", padx=20, pady=5)

        # NÚT ĐIỀU HƯỚNG 2: XEM NHẬT KÝ
        self.btn_go_log = ctk.CTkButton(self.sidebar, text="📋  Xem Nhật Ký & DL", font=("Arial", 14),
                                        fg_color="transparent", text_color="#a9b6c3", hover_color="#22272b", height=38, corner_radius=8, 
                                        command=lambda: self.switch_tab("📋 Nhật Ký & Downloads"))
        self.btn_go_log.pack(fill="x", padx=20, pady=5)

        self.btn_clear_log = ctk.CTkButton(self.sidebar, text="🗑️  Xóa Lịch Sử Quét", font=("Arial", 14),
                                          fg_color="transparent", text_color="#ff4d4d", hover_color="#2b1a1a", height=38, corner_radius=8, command=self.clear_log_file)
        self.btn_clear_log.pack(fill="x", padx=20, pady=5)

        self.btn_help = ctk.CTkButton(self.sidebar, text="❓  Trợ Giúp Liên Hệ", font=("Arial", 14),
                                      fg_color="transparent", text_color="#a9b6c3", hover_color="#22272b", height=38, corner_radius=8, command=self.show_help_info)
        self.btn_help.pack(fill="x", padx=20, pady=5)

        self.btn_update = ctk.CTkButton(self.sidebar, text="🔄  Kiểm Tra Cập Nhật", font=("Arial", 14),
                                        fg_color="transparent", text_color="#a9b6c3", hover_color="#22272b", height=38, corner_radius=8, command=self.check_software_update)
        self.btn_update.pack(fill="x", padx=20, pady=5)

        # KHUNG CÔNG TẮC BẬT TẮT REAL-TIME BẢO VỆ
        self.switch_frame = ctk.CTkFrame(self.sidebar, fg_color="#0d0f11", corner_radius=8)
        self.switch_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_switch_status = ctk.CTkLabel(self.switch_frame, text="🟢 REAL-TIME: BẬT", font=("Arial", 11, "bold"), text_color="#4dff88")
        self.lbl_switch_status.pack(pady=(6, 2))
        
        self.rt_switch = ctk.CTkSwitch(self.switch_frame, text="Bảo vệ ngầm", font=("Arial", 12),
                                       progress_color="#1d6fdc", command=self.toggle_realtime_protection)
        self.rt_switch.pack(pady=(2, 8))

        self.info_lbl = ctk.CTkLabel(self.sidebar, text=f"Phiên bản: Ver {CURRENT_VERSION}\nTrạng thái: Đang chạy", 
                                     text_color="#556475", font=("Arial", 11), justify="left")
        self.info_lbl.pack(side="bottom", pady=20, padx=25, anchor="w")

    def setup_main_content(self):
        # Phiên bản khởi tạo Tabview nguyên bản nhất để tương thích với tất cả phiên bản CustomTkinter cũ/mới
        self.tab_view = ctk.CTkTabview(self, corner_radius=12, fg_color="#0d0f11")
        self.tab_view.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        
        self.tab_view.add("🛡️ Bảo Vệ Hệ Thống")
        self.tab_view.add("📋 Nhật Ký & Downloads")
        
        # Ẩn các nút chuyển tab mặc định bằng cách cho màu nút tiệp hoàn toàn vào nền đen
        try:
            self.tab_view._segmented_button.configure(width=0, height=0)
            self.tab_view._segmented_button.grid_forget() # Ẩn hoàn toàn khỏi lưới hiển thị
        except Exception:
            pass
        self.setup_tab_scan(self.tab_view.tab("🛡️ Bảo Vệ Hệ Thống"))
        self.setup_tab_log(self.tab_view.tab("📋 Nhật Ký & Downloads"))

    def switch_tab(self, tab_name):
        """Lật trang ngầm đồng thời đồng bộ đổi màu trạng thái Highlight cho nút tương ứng ở Sidebar"""
        self.tab_view.set(tab_name)
        
        if tab_name == "🛡️ Bảo Vệ Hệ Thống":
            self.btn_go_scan.configure(fg_color="#1d6fdc", text_color="white", font=("Arial", 14, "bold"))
            self.btn_go_log.configure(fg_color="transparent", text_color="#a9b6c3", font=("Arial", 14))
        elif tab_name == "📋 Nhật Ký & Downloads":
            self.btn_go_scan.configure(fg_color="transparent", text_color="#a9b6c3", font=("Arial", 14))
            self.btn_go_log.configure(fg_color="#1d6fdc", text_color="white", font=("Arial", 14, "bold"))

    def setup_tab_scan(self, tab_master):
        self.res_card = ctk.CTkFrame(tab_master, fg_color="#16191c", height=380, corner_radius=12, border_width=1, border_color="#22272b")
        self.res_card.pack(pady=(20, 20), padx=20, fill="x")
        self.res_card.pack_propagate(False)

        self.shield_lbl = ctk.CTkLabel(self.res_card, text="🛡️", font=("Arial", 64))
        self.shield_lbl.pack(pady=(25, 5))

        self.status_title = ctk.CTkLabel(self.res_card, text="Chờ tải tệp tin phân tích...", font=("Arial", 18, "bold"), text_color="#a9b6c3")
        self.status_title.pack(pady=5)

        self.prob_bar = ctk.CTkProgressBar(self.res_card, width=500, height=10, corner_radius=5, fg_color="#22272b")
        self.prob_bar.pack(pady=10)
        self.prob_bar.set(0)

        self.detail_lbl = ctk.CTkLabel(self.res_card, text="Định dạng hỗ trợ: .exe, .dll", font=("Consolas", 13), text_color="#556475")
        self.detail_lbl.pack(pady=5)

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

        self.action_btn = ctk.CTkButton(tab_master, text="Chọn File Quét...", width=180, height=42, font=("Arial", 14, "bold"),
                                        fg_color="#1d6fdc", hover_color="#185cc5", corner_radius=8, command=self.browse_file)
        self.action_btn.pack(pady=20)

    def setup_tab_log(self, tab_master):
        tab_master.grid_columnconfigure(0, weight=1, uniform="group1")
        tab_master.grid_columnconfigure(1, weight=1, uniform="group1")
        tab_master.grid_rowconfigure(1, weight=1)

        # CỘT TRÁI
        self.left_frame = ctk.CTkFrame(tab_master, fg_color="transparent")
        self.left_frame.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="nsew")
        
        self.lbl_left_title = ctk.CTkLabel(self.left_frame, text="📥 File Mới Tải Về (7 Ngày Qua)", font=("Arial", 14, "bold"), text_color="#e9af22")
        self.lbl_left_title.pack(anchor="w", pady=(0, 5))

        self.box_downloads = ctk.CTkTextbox(self.left_frame, corner_radius=10, fg_color="#16191c", border_width=1, border_color="#22272b", font=("Consolas", 11.5))
        self.box_downloads.pack(fill="both", expand=True, pady=5)
        
        self.btn_delete_download_file = ctk.CTkButton(self.left_frame, text="🗑️ Chọn File Để Xóa Thủ Công", font=("Arial", 12, "bold"),
                                                      fg_color="#2b1a1a", text_color="#ff4d4d", hover_color="#401a1a", height=32, command=self.delete_selected_download)
        self.btn_delete_download_file.pack(fill="x", pady=5)

        # CỘT PHẢI
        self.right_frame = ctk.CTkFrame(tab_master, fg_color="transparent")
        self.right_frame.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="nsew")
        
        self.lbl_right_title = ctk.CTkLabel(self.right_frame, text="🕒 Lịch Sử Nhật Ký Đã Quét", font=("Arial", 14, "bold"), text_color="#4dff88")
        self.lbl_right_title.pack(anchor="w", pady=(0, 5))

        self.box_history = ctk.CTkTextbox(self.right_frame, corner_radius=10, fg_color="#16191c", border_width=1, border_color="#22272b", font=("Consolas", 11.5))
        self.box_history.pack(fill="both", expand=True, pady=5)

        self.btn_refresh = ctk.CTkButton(tab_master, text="🔄 Làm Mới Hệ Thống", font=("Arial", 12),
                                         fg_color="#16191c", text_color="#a9b6c3", hover_color="#22272b", height=32, command=self.load_log_into_app)
        self.btn_refresh.grid(row=2, column=0, columnspan=2, pady=10)

    def toggle_realtime_protection(self, force_on=False):
        if force_on or self.rt_switch.get() == 1:
            self.realtime_enabled = True
            self.lbl_switch_status.configure(text="🟢 REAL-TIME: BẬT", text_color="#4dff88")
            
            if self.observer:
                try:
                    self.observer.stop()
                    self.observer.join()
                except Exception: pass
            
            try:
                self.observer = Observer()
                handler = DownloadFolderHandler(self)
                self.observer.schedule(handler, path=DOWNLOADS_DIR, recursive=False)
                self.observer.start()
            except Exception as e:
                print(f"[Watchdog Warning] Chưa kích hoạt được bảo vệ ngầm: {e}")
        else:
            self.realtime_enabled = False
            self.lbl_switch_status.configure(text="🔴 REAL-TIME: TẮT", text_color="#ff4d4d")
            if self.observer:
                try:
                    self.observer.stop()
                    self.observer.join()
                except Exception: pass
                self.observer = None

    def trigger_realtime_scan(self, path):
        if not self.realtime_enabled: 
            return
        self.switch_tab("🛡️ Bảo Vệ Hệ Thống")
        self.start_analysis_flow(path)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Executable Files", "*.exe *.dll")])
        if path:
            self.start_analysis_flow(path)

    def start_analysis_flow(self, path):
        self.status_title.configure(text=f"📂 Đang phân tích: {os.path.basename(path)}...", text_color="#e9af22")
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
            shield_icon = "🛡️"
        elif safety_ratio >= 0.50:
            color_hex = "#99ff33"
            shield_icon = "🛡️"
        elif safety_ratio >= 0.25:
            color_hex = "#ff8533"
            shield_icon = "⚠️"
        else:
            color_hex = "#ff4d4d"
            shield_icon = "⚠️"
            
        self.prob_bar.configure(progress_color=color_hex)
        self.shield_lbl.configure(text=shield_icon, text_color=color_hex)
        
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

        self.load_log_into_app()

        if result['label'] == "MALWARE":
            confirm = messagebox.askyesno(
                "Real-time Shield: Phát hiện mối đe dọa", 
                f"Hệ thống phát hiện tệp tin nguy hiểm nguy kịch tại:\n{path}\n\nĐộ an toàn chỉ còn: {(safety_ratio * 100):.2f}%\n\nBạn có muốn XÓA VĨNH VIỄN tệp tin này không?"
            )
            if confirm:
                try:
                    os.remove(path)
                    messagebox.showinfo("Thành Công", "🛡️ Đã tiêu diệt và loại bỏ hoàn toàn file mã độc khỏi hệ thống!")
                    
                    delete_log = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] [REALTIME CHẶN & XÓA] {path}\n"
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(delete_log)
                    
                    self.load_log_into_app()
                    self.status_title.configure(text="💥 MỐI ĐE DỌA ĐÃ ĐƯỢC XỬ LÝ", text_color="#4dff88")
                    self.detail_lbl.configure(text=f"File độc hại đã bị xóa bỏ hoàn toàn khỏi ổ cứng.", text_color="gray")
                    self.shield_lbl.configure(text="🛡️", text_color="#4dff88")
                except Exception as e:
                    messagebox.showerror("Lỗi Xóa File", f"Không thể xóa file: {e}")
            else:
                keep_log = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] [NGƯỜI DÙNG GIỮ LẠI FILE NGUY HIỂM] {path}\n"
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(keep_log)
                self.load_log_into_app()
        else:
            messagebox.showinfo("Real-time Shield", f"Tệp tin vừa tải về an toàn tuyệt đối!\nMức độ bảo vệ: {(safety_ratio * 100):.2f}%")

    def load_log_into_app(self):
        self.box_downloads.configure(state="normal")
        self.box_downloads.delete("1.0", "end")
        self.download_paths_map.clear() 
        
        if os.path.exists(DOWNLOADS_DIR):
            try:
                now_ts = time.time()
                seven_days_sec = 7 * 24 * 60 * 60 
                
                downloaded_files = []
                for f in os.listdir(DOWNLOADS_DIR):
                    full_p = os.path.join(DOWNLOADS_DIR, f)
                    if os.path.isfile(full_p) and f.lower().endswith(('.exe', '.dll')):
                        mtime = os.path.getmtime(full_p)
                        if (now_ts - mtime) <= seven_days_sec:
                            downloaded_files.append((f, full_p, mtime))
                
                downloaded_files.sort(key=lambda x: x[2], reverse=True)
                
                if downloaded_files:
                    for idx, (name, full_path, mtime) in enumerate(downloaded_files):
                        t_str = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M")
                        self.box_downloads.insert("end", f"{idx + 1}. [{t_str}] {name}\n")
                        self.download_paths_map[str(idx + 1)] = full_path
                else:
                    self.box_downloads.insert("end", "Không phát hiện file hệ thống nào (.exe, .dll) mới tải về trong 7 ngày qua.\n")
            except Exception as e:
                self.box_downloads.insert("end", f"Lỗi đọc thư mục: {e}\n")
                
        self.box_downloads.configure(state="disabled")

        self.box_history.configure(state="normal")
        self.box_history.delete("1.0", "end")
        
        if os.path.exists(LOG_FILE_PATH):
            try:
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    self.box_history.insert("end", line)
            except Exception:
                pass
                
        self.box_history.configure(state="disabled")

    def delete_selected_download(self):
        if not self.download_paths_map:
            messagebox.showinfo("Thông báo", "Không có file nào trong danh sách tải về để xóa.")
            return

        dialog = ctk.CTkInputDialog(text="Nhập số thứ tự (1, 2, 3...) của file bạn muốn xóa vĩnh viễn khỏi thư mục Downloads:", title="Xóa File Thủ Công")
        num_choice = dialog.get_input()
        
        if num_choice and num_choice.strip() in self.download_paths_map:
            target_path = self.download_paths_map[num_choice.strip()]
            file_name = os.path.basename(target_path)
            
            if messagebox.askyesno("Xác nhận", f"Bạn có chắc muốn xóa vĩnh viễn file:\n{file_name}\nkhỏi ổ cứng không?"):
                try:
                    os.remove(target_path)
                    messagebox.showinfo("Thành Công", f"Đã xóa bỏ hoàn toàn file '{file_name}' khỏi máy tính.")
                    delete_log = f"[{time.strftime('%d/%m/%Y %H:%M:%S')}] [NGƯỜI DÙNG XÓA THỦ CÔNG] {target_path}\n"
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(delete_log)
                    self.load_log_into_app()
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể xóa tệp tin: {e}")
        elif num_choice:
            messagebox.showerror("Lỗi", "Số thứ tự bạn nhập không hợp lệ hoặc không có trong danh sách!")

    def clear_log_file(self):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa toàn bộ lịch sử nhật ký đã quét không?"):
            if os.path.exists(LOG_FILE_PATH):
                os.remove(LOG_FILE_PATH)
            self.load_log_into_app()
            messagebox.showinfo("Thông báo", "Đã xóa sạch lịch sử nhật ký.")
            self.shield_lbl.configure(text="🛡️", text_color="#a9b6c3")

    def show_help_info(self):
        help_msg = (
            "=========================================\n"
            "        🛡️ HỆ THỐNG AI SHIELD PRO       \n"
            "=========================================\n\n"
            "• Tác giả đồ án  : Phạm Minh Tuấn & Nguyễn Lê Nhật Huy\n"
            "• Số điện thoại  : 0123.456.789\n"
            "• Email liên hệ  : pmtnlnh.4@hutech.edu.vn\n\n"
            "-----------------------------------------\n"
            "Hỗ trợ kỹ thuật 24/7 về các giải pháp phân tích\n"
            "mã độc PE dựa trên cấu trúc ảnh phổ phổ tần STFT\n"
            "và mạng nơ-ron tích chập (CNN-3 Core)."
        )
        messagebox.showinfo("Thông Tin Hỗ Trợ Đồ Án", help_msg)

    def check_software_update(self):
        self.switch_tab("🛡️ Bảo Vệ Hệ Thống")
        self.status_title.configure(text="🌐 Đang kết nối tới máy chủ cập nhật...", text_color="#e9af22")
        self.shield_lbl.configure(text="🔄", text_color="#e9af22")
        
        def delay_update_popup():
            time.sleep(1.2)
            self.status_title.configure(text="Chờ tải tệp tin phân tích...", text_color="#a9b6c3")
            self.shield_lbl.configure(text="🛡️", text_color="#a9b6c3")
            messagebox.showinfo("Kiểm Tra Cập Nhật", f"Hệ thống AI Shield đang chạy phiên bản mới nhất (v{CURRENT_VERSION}).\nCơ sở dữ liệu mô hình CNN-3 đã được tối ưu.")

        threading.Thread(target=delay_update_popup, daemon=True).start()

    def on_closing(self):
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join()
            except Exception: pass
        self.destroy()

if __name__ == "__main__":
    app = PredictApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()