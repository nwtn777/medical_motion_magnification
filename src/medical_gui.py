import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from collections import deque
from medical_optimized_concurrent_futures import Magnify, detectar_rostro_y_roi, inicializar_camara, calcular_bpm, liberar_recursos

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_image))

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        self.line1, = self.ax1.plot([], [], label="Intensidad promedio")
        self.line2, = self.ax2.plot([], [], label="Pulso (BPM)", color='r')
        self.ax1.set_ylim(0, 255)
        self.ax1.set_xlim(0, 300)
        self.ax1.set_title("Señal promedio ROI (Pulso)")
        self.ax1.set_xlabel("Frame")
        self.ax1.set_ylabel("Intensidad promedio")
        self.ax1.legend()
        self.ax2.set_ylim(40, 180)
        self.ax2.set_xlim(0, 300)
        self.ax2.set_title("Pulso cardiaco estimado (BPM)")
        self.ax2.set_xlabel("Frame")
        self.ax2.set_ylabel("BPM")
        self.ax2.legend()
        self.fig.tight_layout()

    def update_plot(self, signal_buffer, bpm_buffer):
        self.line1.set_ydata(signal_buffer)
        self.line1.set_xdata(np.arange(len(signal_buffer)))
        self.ax1.set_xlim(0, max(300, len(signal_buffer)))
        self.ax1.set_ylim(min(signal_buffer, default=0), max(signal_buffer, default=255))
        self.line2.set_ydata(bpm_buffer)
        self.line2.set_xdata(np.arange(len(bpm_buffer)))
        self.ax2.set_xlim(0, max(300, len(bpm_buffer)))
        self.ax2.set_ylim(40, 180)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnificación Médica - GUI")
        self.cam, self.img1 = inicializar_camara()
        if self.cam is None or self.img1 is None:
            raise RuntimeError("No se pudo inicializar la cámara.")
        self.roi = detectar_rostro_y_roi(self.img1)
        if self.roi is None:
            liberar_recursos(self.cam)
            raise RuntimeError("No se detectó rostro.")
        x, y, w_roi, h_roi = self.roi
        gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h_roi, x:x+w_roi]
        self.fps = 20
        self.alpha = 200
        self.lambda_c = 20
        self.fl = 0.5
        self.fh = 3.0
        self.s = Magnify(roi_gray, self.alpha, self.lambda_c, self.fl, self.fh, self.fps)
        self.prev_gray = gray.copy()
        self.signal_buffer = deque(maxlen=300)
        self.bpm_buffer = deque(maxlen=300)
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000/self.fps))

    def init_ui(self):
        self.video_widget = VideoWidget(self)
        self.plot_canvas = PlotCanvas(self)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setMinimum(1)
        self.slider_alpha.setMaximum(500)
        self.slider_alpha.setValue(self.alpha)
        self.slider_alpha.valueChanged.connect(self.change_alpha)
        self.slider_lambda = QSlider(Qt.Horizontal)
        self.slider_lambda.setMinimum(1)
        self.slider_lambda.setMaximum(100)
        self.slider_lambda.setValue(self.lambda_c)
        self.slider_lambda.valueChanged.connect(self.change_lambda)
        self.label_alpha = QLabel(f"Alpha: {self.alpha}")
        self.label_lambda = QLabel(f"Lambda: {self.lambda_c}")
        self.btn_quit = QPushButton("Salir")
        self.btn_quit.clicked.connect(self.close)
        layout = QGridLayout()
        layout.addWidget(self.video_widget, 0, 0, 1, 2)
        layout.addWidget(self.plot_canvas, 1, 0, 1, 2)
        layout.addWidget(self.label_alpha, 2, 0)
        layout.addWidget(self.slider_alpha, 2, 1)
        layout.addWidget(self.label_lambda, 3, 0)
        layout.addWidget(self.slider_lambda, 3, 1)
        layout.addWidget(self.btn_quit, 4, 0, 1, 2)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def change_alpha(self, value):
        self.alpha = value
        self.label_alpha.setText(f"Alpha: {self.alpha}")
        self.s.alpha = self.alpha

    def change_lambda(self, value):
        self.lambda_c = value
        self.label_lambda.setText(f"Lambda: {self.lambda_c}")
        self.s.lambda_c = self.lambda_c
        self.s.delta = self.lambda_c / 8. / (1 + self.alpha)

    def update(self):
        x, y, w_roi, h_roi = self.roi
        ret, final_img = self.cam.read()
        if not ret or final_img is None:
            return
        roi_img = final_img[y:y+h_roi, x:x+w_roi]
        mask = np.zeros_like(roi_img)
        mask[..., 1] = 255
        gray_prev_roi = self.prev_gray[y:y+h_roi, x:x+w_roi]
        out = self.s.Magnify(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))
        final_img[y:y+h_roi, x:x+w_roi] = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        self.video_widget.update_frame(final_img)
        mean_signal = np.mean(out)
        self.signal_buffer.append(mean_signal)
        bpm = calcular_bpm(self.signal_buffer, self.fps)
        self.bpm_buffer.append(bpm)
        self.plot_canvas.update_plot(self.signal_buffer, self.bpm_buffer)
        self.prev_gray[y:y+h_roi, x:x+w_roi] = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    def closeEvent(self, event):
        liberar_recursos(self.cam)
        plt.close('all')
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
