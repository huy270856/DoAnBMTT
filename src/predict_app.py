import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Tk
from tkinter import scrolledtext
import threading

MODEL_PATH = r"C:\Huy_Malware_AI_v2\models\malware_spectrogram_model.keras"
TEMP_IMG_PATH = r"C:\Huy_Malware_AI_v2\src\temp_predict.png"

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
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
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

            ax.specgram(signal, NFFT=NFFT_WINDOW, Fs=SAMPLING_RATE, 
                        window=np.hanning(NFFT_WINDOW), noverlap=NOVERLAP, cmap='inferno')

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
            return None, "Không thể tạo spectrogram"
        
        try:
            img = image.load_img(TEMP_IMG_PATH, target_size=TARGET_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            if os.path.exists(TEMP_IMG_PATH):
                os.remove(TEMP_IMG_PATH)
                
            if prediction > 0.5:
                label = "MALWARE"
                malware_prob = prediction * 100
                benign_prob = (1 - prediction) * 100
                confidence = malware_prob
            else:
                label = "BENIGN"
                malware_prob = prediction * 100
                benign_prob = (1 - prediction) * 100
                confidence = benign_prob
                
            return {
                'label': label,
                'confidence': confidence,
                'malware_prob': malware_prob,
                'benign_prob': benign_prob
            }, "OK"
        except Exception as e:
            return None, str(e)

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Malware Spectrogram Classifier - MSIC")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        try:
            self.predictor = MalwarePredictor()
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Lỗi", str(e))
            self.root.destroy()
            return
        
        self.setup_ui()
        
    def setup_ui(self):
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill='x')
        
        title_label = ttk.Label(title_frame, text="MALWARE SPECTROGRAM CLASSIFIER", font=('Helvetica', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="MSIC Framework - CNN-based Detection", font=('Helvetica', 10))
        subtitle_label.pack()
        
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        file_frame = ttk.LabelFrame(self.root, text="Chọn File", padding="10")
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, font=('Courier', 10))
        file_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side='left', padx=(0, 5))
        
        predict_btn = ttk.Button(file_frame, text="Phân Tích", command=self.start_prediction)
        predict_btn.pack(side='left')
        
        progress_frame = ttk.Frame(self.root, padding="10")
        progress_frame.pack(fill='x', padx=10)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill='x')
        
        result_frame = ttk.LabelFrame(self.root, text="Kết Quả", padding="10")
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, font=('Courier', 10))
        self.result_text.pack(fill='both', expand=True)
        
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="Sẵn sàng")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief='sunken', padding=5)
        status_label.pack(fill='x')
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file",
            filetypes=[("Executable files", "*.exe *.dll *.bat *.ps1"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def start_prediction(self):
        file_path = self.file_path_var.get().strip('"')
        
        if not file_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Lỗi", "File không tồn tại")
            return
        
        self.predict_btn_state(False)
        self.progress.start(10)
        self.status_var.set("Đang phân tích...")
        
        thread = threading.Thread(target=self.run_prediction, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def run_prediction(self, file_path):
        try:
            result, status = self.predictor.predict(file_path)
            
            if status == "OK":
                self.root.after(0, self.display_result, file_path, result)
            else:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", status))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, self.prediction_complete)
    
    def display_result(self, file_path, result):
        self.result_text.delete(1.0, 'end')
        
        self.result_text.insert('end', "="*60 + "\n", 'header')
        self.result_text.insert('end', f"File: {os.path.basename(file_path)}\n")
        self.result_text.insert('end', f"Path: {file_path}\n")
        self.result_text.insert('end', "="*60 + "\n\n")
        
        self.result_text.insert('end', "KẾT QUẢ PHÂN TÍCH:\n", 'bold')
        self.result_text.insert('end', "-"*60 + "\n")
        
        if result['label'] == "MALWARE":
            self.result_text.insert('end', f"NHÃN: {result['label']}\n", 'malware')
        else:
            self.result_text.insert('end', f"NHÃN: {result['label']}\n", 'benign')
        
        self.result_text.insert('end', f"Độ tin cậy: {result['confidence']:.2f}%\n\n")
        
        self.result_text.insert('end', "XÁC SUẤT:\n", 'bold')
        self.result_text.insert('end', "-"*60 + "\n")
        self.result_text.insert('end', f" Malware:  {result['malware_prob']:>6.2f}%\n")
        self.result_text.insert('end', f" Benign:   {result['benign_prob']:>6.2f}%\n\n")
        
        self.result_text.insert('end', "KHUYẾN NGHỊ:\n", 'bold')
        self.result_text.insert('end', "-"*60 + "\n")
        if result['label'] == "MALWARE":
            if result['confidence'] > 90:
                self.result_text.insert('end', " Nguy hiểm! KHÔNG NÊN thực thi.\n")
            else:
                self.result_text.insert('end', " Đáng ngờ. Cần kiểm tra thêm.\n")
        else:
            if result['confidence'] > 90:
                self.result_text.insert('end', " An toàn.\n")
            else:
                self.result_text.insert('end', " Có vẻ benign nhưng độ tin cậy thấp.\n")
        
        self.result_text.insert('end', "\n" + "="*60 + "\n")
        
        self.result_text.tag_configure('header', font=('Courier', 10, 'bold'))
        self.result_text.tag_configure('bold', font=('Courier', 10, 'bold'))
        self.result_text.tag_configure('malware', foreground='red', font=('Courier', 10, 'bold'))
        self.result_text.tag_configure('benign', foreground='green', font=('Courier', 10, 'bold'))
    
    def predict_btn_state(self, enabled):
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Button):
                        child.config(state='normal' if enabled else 'disabled')
    
    def prediction_complete(self):
        self.progress.stop()
        self.status_var.set("Hoàn tất")
        self.predict_btn_state(True)

def main():
    root = Tk()
    app = PredictApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()