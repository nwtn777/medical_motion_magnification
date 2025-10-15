import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from medical_optimized_concurrent_futures import Magnify
from PIL import Image, ImageTk

class MedicalMotionMagnificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Motion Magnification")
        self.root.geometry("1200x800")

        # Variables
        self.video_source = None
        self.cap = None
        self.is_processing = False
        self.alpha = tk.DoubleVar(value=10)
        self.lambda_c = tk.DoubleVar(value=16)
        self.fl = tk.DoubleVar(value=0.05)
        self.fh = tk.DoubleVar(value=0.4)
        self.sampling_rate = tk.DoubleVar(value=30)

        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create left panel for controls
        left_panel = ttk.LabelFrame(main_frame, text="Controls")
        left_panel.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        # Parameters
        ttk.Label(left_panel, text="Alpha (amplification):").pack(pady=5)
        ttk.Entry(left_panel, textvariable=self.alpha).pack(pady=5)

        ttk.Label(left_panel, text="Lambda_c (cutoff wavelength):").pack(pady=5)
        ttk.Entry(left_panel, textvariable=self.lambda_c).pack(pady=5)

        ttk.Label(left_panel, text="fl (low frequency):").pack(pady=5)
        ttk.Entry(left_panel, textvariable=self.fl).pack(pady=5)

        ttk.Label(left_panel, text="fh (high frequency):").pack(pady=5)
        ttk.Entry(left_panel, textvariable=self.fh).pack(pady=5)

        ttk.Label(left_panel, text="Sampling Rate (fps):").pack(pady=5)
        ttk.Entry(left_panel, textvariable=self.sampling_rate).pack(pady=5)

        # Buttons
        ttk.Button(left_panel, text="Load Video", command=self.load_video).pack(pady=20)
        self.process_button = ttk.Button(left_panel, text="Start Processing", command=self.toggle_processing)
        self.process_button.pack(pady=5)
        ttk.Button(left_panel, text="Save Video", command=self.save_video).pack(pady=20)

        # Create right panel for video display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Original video canvas
        self.original_canvas = tk.Canvas(right_panel, width=400, height=300)
        self.original_canvas.pack(pady=5)
        ttk.Label(right_panel, text="Original Video").pack()

        # Processed video canvas
        self.processed_canvas = tk.Canvas(right_panel, width=400, height=300)
        self.processed_canvas.pack(pady=5)
        ttk.Label(right_panel, text="Processed Video").pack()

    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.video_source = file_path
            self.cap = cv2.VideoCapture(self.video_source)
            messagebox.showinfo("Success", "Video loaded successfully!")

    def toggle_processing(self):
        if not self.video_source:
            messagebox.showerror("Error", "Please load a video first!")
            return

        if self.is_processing:
            self.is_processing = False
            self.process_button.config(text="Start Processing")
        else:
            self.is_processing = True
            self.process_button.config(text="Stop Processing")
            self.process_video()

    def process_video(self):
        if not self.is_processing:
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize magnification if this is the first frame
            if not hasattr(self, 'magnifier'):
                self.magnifier = Magnify(
                    gray,
                    self.alpha.get(),
                    self.lambda_c.get(),
                    self.fl.get(),
                    self.fh.get(),
                    self.sampling_rate.get()
                )
            
            # Process the frame
            processed = self.magnifier.Magnify(gray)
            
            # Convert processed result to uint8
            processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
            processed = np.uint8(processed)
            
            # Display original and processed frames
            self.display_frame(self.original_canvas, frame)
            self.display_frame(self.processed_canvas, cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
            
            # Schedule next frame processing
            self.root.after(int(1000/self.sampling_rate.get()), self.process_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to beginning
            self.process_video()

    def display_frame(self, canvas, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (400, 300))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.photo = photo

    def save_video(self):
        if not hasattr(self, 'magnifier'):
            messagebox.showerror("Error", "No processed video to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if file_path:
            # Implementation of video saving logic goes here
            messagebox.showinfo("Success", "Video saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalMotionMagnificationGUI(root)
    root.mainloop()
