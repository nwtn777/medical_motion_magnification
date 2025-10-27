# Medical Pulse Magnifier

## Overview
The Medical Pulse Magnifier is a Python project designed to process video frames in real-time to magnify pulse signals from a user's forehead. Utilizing advanced image processing techniques, this application captures video input, detects facial features, and applies magnification to the region of interest (ROI) to visualize pulse changes.

## Features
- Real-time video processing using OpenCV.
- Optical flow visualization to track motion.
- Laplacian pyramid-based image magnification.
- Graphical representation of average intensity and estimated heart rate (BPM).

## Installation
To set up the project, ensure you have Python installed on your machine. Then, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/medical-pulse-magnifier.git
   cd medical-pulse-magnifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command in your terminal:
```
python src/medical_optimized_concurrent_futures.py
```

Make sure your camera is connected and accessible. The application will open a window displaying the video feed with the magnified pulse ROI.

## Requirements
The project requires the following Python packages:
- opencv-python
- scipy
- scikit-image
- numpy
- matplotlib
- pyrtools
- PyQt5

el proyecto todavía muestra problemas de inconsistencia de medición del pulso de BPM


## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.