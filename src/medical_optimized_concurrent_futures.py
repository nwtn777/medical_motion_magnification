import cv2
import scipy.signal as signal
import scipy.fftpack as fftpack
from skimage import img_as_float, img_as_ubyte
import numpy as np
import time
import pyrtools as pt
import copy
import concurrent.futures
import matplotlib.pyplot as plt
from collections import deque

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    cv2.putText(vis, "magnitud_prom: " + str(np.mean(magnitude)) + "px/f", [50,450], 1, 2, (0, 0, 255))
    return vis

def reconPyr(pyr):
    filt2 = pt.binomial_filter(5)
    maxLev = len(pyr)
    levs = range(0, maxLev)
    res = []
    for lev in range(maxLev-1, -1, -1):
        if lev in levs and len(res) == 0:
            res = pyr[lev]
        elif len(res) != 0:
            res_sz = res.shape
            new_sz = pyr[lev].shape
            if res_sz[0] == 1:
                hi2 = pt.upConv(image=res, filt=filt2, step=(2,1), stop=(new_sz[1], new_sz[0])).T
            elif res_sz[1] == 1:
                hi2 = pt.upConv(image=res, filt=filt2.T, step=(1,2), stop=(new_sz[1], new_sz[0])).T
            else:
                hi = pt.upConv(image=res, filt=filt2, step=(2,1), stop=(new_sz[0], res_sz[1]))
                hi2 = pt.upConv(image=hi, filt=filt2.T, step=(1,2), stop=(new_sz[0], new_sz[1]))
            if lev in levs:
                bandIm = pyr[lev]
                res = hi2 + bandIm
            else:
                res = hi2
    return res

class Magnify(object):
    def __init__(self, gray1, alpha, lambda_c, fl, fh, samplingRate):
        [low_a, low_b] = signal.butter(1, fl/samplingRate, 'low')
        [high_a, high_b] = signal.butter(1, fh/samplingRate, 'low')
        py1 = pt.pyramids.LaplacianPyramid(gray1)
        py1._build_pyr()
        pyramid_1 = py1.pyr_coeffs
        nLevels = len(pyramid_1)
        self.filtered = pyramid_1
        self.alpha = alpha
        self.fl = fl
        self.fh = fh
        self.samplingRate = samplingRate
        self.low_a = low_a
        self.low_b = low_b
        self.high_a = high_a
        self.high_b = high_b
        self.width = gray1.shape[0]
        self.height = gray1.shape[1]
        self.gray1 = img_as_float(gray1)
        self.lowpass1 = copy.deepcopy(pyramid_1)
        self.lowpass2 = copy.deepcopy(self.lowpass1)
        self.pyr_prev = copy.deepcopy(pyramid_1)
        self.filtered = [None for _ in range(nLevels)]
        self.nLevels = nLevels
        self.lambd = (self.width**2 + self.height**2) / 3.
        self.lambda_c = lambda_c
        self.delta = self.lambda_c / 8. / (1 + self.alpha)

    def Magnify(self, gray2):
        gray2 = img_as_float(gray2)
        py2 = pt.pyramids.LaplacianPyramid(gray2)
        py2._build_pyr()
        pyr = py2.pyr_coeffs
        nLevels = self.nLevels
        for u in range(nLevels):
            self.lowpass1[(u,0)] = (-self.high_b[1]*self.lowpass1[(u,0)] + self.high_a[0]*pyr[(u,0)] + self.high_a[1]*self.pyr_prev[(u,0)]) / self.high_b[0]
            self.lowpass2[(u,0)] = (-self.low_b[1]*self.lowpass2[(u,0)] + self.low_a[0]*pyr[(u,0)] + self.low_a[1]*self.pyr_prev[(u,0)]) / self.low_b[0]
            self.filtered[u] = self.lowpass1[(u,0)] - self.lowpass2[(u,0)]
        self.pyr_prev = copy.deepcopy(pyr)
        exaggeration_factor = 2
        lambd = self.lambd
        delta = self.delta
        filtered = self.filtered
        for l in range(nLevels-1, -1, -1):
            currAlpha = lambd / delta / 8. - 1
            currAlpha = currAlpha * exaggeration_factor
            if (l == nLevels - 1 or l == 0):
                filtered[l] = np.zeros(np.shape(filtered[l]))
            elif (currAlpha > self.alpha):
                filtered[l] = self.alpha * filtered[l]
            else:
                filtered[l] = currAlpha * filtered[l]
            lambd = lambd / 2.
        output = reconPyr(filtered)
        output = gray2 + output
        output[output < 0] = 0
        output[output > 1] = 1
        output = img_as_ubyte(output)
        return output

def process_frame(final_img, prev_gray, mask, s):
    gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    out = s.Magnify(gray)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, out, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    mag_fft_shift = np.fft.fftshift(np.abs(np.fft.fftshift(magnitude)))
    return out, flow, final_img, rgb, mag_fft_shift, gray

if __name__ == "__main__":
    fps = 20
    alpha = 100
    lambda_c = 20
    fl = 0.8
    fh = 2.0

    cam = cv2.VideoCapture(0)
    ret, img1 = cam.read()
    if not ret or img1 is None:
        print("Error: Could not read from camera.")
        exit(1)

    # --- Detección automática de rostro y ROI de la frente ---
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_img1, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print("No se detectó ningún rostro. Ajusta la iluminación o posiciónate frente a la cámara.")
        exit(1)

    # Tomar el primer rostro detectado
    (x_face, y_face, w_face, h_face) = faces[0]

    # Definir ROI de la frente (porción superior del rostro)
    x = x_face + int(w_face * 0.25)
    y = y_face + int(h_face * 0.10)
    w_roi = int(w_face * 0.5)
    h_roi = int(h_face * 0.18)
    roi = (x, y, w_roi, h_roi)

    # Visual feedback (opcional)
    img1_roi = img1.copy()
    cv2.rectangle(img1_roi, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 2)
    cv2.imshow("ROI Frente Detectado", img1_roi)
    cv2.waitKey(1000)
    cv2.destroyWindow("ROI Frente Detectado")
    # --------------------------------------------------------

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h_roi, x:x+w_roi]
    s = Magnify(roi_gray, alpha, lambda_c, fl, fh, fps)
    prev_gray = gray

    signal_buffer = deque(maxlen=300)
    bpm_buffer = deque(maxlen=300)  # Nuevo buffer para BPM

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    (line1,) = ax1.plot([], [], label="Intensidad promedio")
    (line2,) = ax2.plot([], [], label="Pulso (BPM)", color='r')
    ax1.set_ylim(0, 255)
    ax1.set_xlim(0, 300)
    ax1.set_title("Señal promedio ROI (Pulso)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Intensidad promedio")
    ax2.set_ylim(40, 180)
    ax2.set_xlim(0, 300)
    ax2.set_title("Pulso cardiaco estimado (BPM)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("BPM")
    ax2.legend()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            t1 = time.perf_counter()
            ret, final_img = cam.read()
            if not ret or final_img is None:
                print("Error: Could not read from camera.")
                break

            # Extraer ROI
            roi_img = final_img[y:y+h_roi, x:x+w_roi]
            mask = np.zeros_like(roi_img)
            mask[..., 1] = 255

            future = executor.submit(process_frame, roi_img, prev_gray[y:y+h_roi, x:x+w_roi], mask, s)
            out, flow, _, rgb, mag_fft_shift, roi_gray = future.result()

            # Mostrar ROI magnificado en la imagen original
            final_img[y:y+h_roi, x:x+w_roi] = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(final_img, (x, y), (x+w_roi, y+h_roi), (0, 255, 0), 2)
            cv2.imshow('Magnified Pulse ROI', final_img)

            # --- Graficar señal promedio ---
            mean_signal = np.mean(out)
            signal_buffer.append(mean_signal)
            line1.set_ydata(signal_buffer)
            line1.set_xdata(np.arange(len(signal_buffer)))
            ax1.set_xlim(0, max(300, len(signal_buffer)))
            ax1.set_ylim(min(signal_buffer, default=0), max(signal_buffer, default=255))
            # --------------------------------

            # --- Calcular pulso cardiaco y graficar ---
            bpm = None
            if len(signal_buffer) > fps * 5:  # Al menos 5 segundos de señal
                signal_array = np.array(signal_buffer)
                detrended = signal_array - np.mean(signal_array)
                freqs = np.fft.rfftfreq(len(detrended), d=1.0/fps)
                fft_mag = np.abs(np.fft.rfft(detrended))
                idx = np.where((freqs > 0.8) & (freqs < 3.0))
                if len(idx[0]) > 0:
                    peak_freq = freqs[idx][np.argmax(fft_mag[idx])]
                    bpm = peak_freq * 60.0
                    bpm_buffer.append(bpm)
                    print(f"{bpm:.1f}")  # Solo imprime el BPM
                else:
                    bpm_buffer.append(np.nan)
            else:
                bpm_buffer.append(np.nan)

            line2.set_ydata(bpm_buffer)
            line2.set_xdata(np.arange(len(bpm_buffer)))
            ax2.set_xlim(0, max(300, len(bpm_buffer)))
            ax2.set_ylim(40, 180)
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            # --------------------------------

            t2 = time.perf_counter()
            if cv2.waitKey(1) & 0xFF == 27:
                break
            prev_gray[y:y+h_roi, x:x+w_roi] = roi_gray
            # if t2 - t1 > 1. / fps:
                # print("delayed")

    cam.release()
    cv2.destroyAllWindows()