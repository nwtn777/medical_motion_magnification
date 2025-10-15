import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    required = [
        "opencv-python",
        "numpy",
        "matplotlib",
        "PyQt5",
        "scipy",
        "scikit-image",
        "pyrtools"
    ]
    for pkg in required:
        try:
            __import__(pkg.split('-')[0])
        except ImportError:
            print(f"Instalando {pkg}...")
            install(pkg)
    gui_path = os.path.join(os.path.dirname(__file__), "medical_gui.py")
    print("Lanzando la GUI...")
    subprocess.run([sys.executable, gui_path])

if __name__ == "__main__":
    main()
