# Pest Identification Web Application

## Overview
This project is a web-based application built with **Flask** and **YOLO (Ultralytics)** for identifying agricultural pests from images. It uses a custom-trained object detection model to detect various pest species, draw bounding boxes around them, and provide detailed information (such as scientific name, typical damage, and control methods) to help farmers and agronomists manage infestations.

## Technical Architecture
- **Backend Framework**: Python with Flask
- **Machine Learning / Object Detection**: YOLOv8 (via `ultralytics` package)
- **Image Processing**: OpenCV (`cv2`) and NumPy
- **Frontend**: HTML/CSS/JS (served from `templates/` and `static/`)
- **Trained Model**: Custom `.pt` weights (`best.pt`) configured with a 16-class YAML file (`ip102.yaml`).

## Features
- **Upload Image**: Interface for users to upload images of agricultural pests.
- **Computer Vision Model**: Leverages a pre-trained `best.pt` YOLO model (or falls back to `yolov8n.pt` if missing) to perform inference.
- **Image Preprocessing**: Dynamically resizes inputs to 640x640 using OpenCV, scales bounding boxes accurately back to original image dimensions, and displays confidence scores.
- **Detailed Insect Information**: Once a specific pest class is detected (e.g., "Rice Leaf Roller", "Asiatic Rice Borer"), the backend returns contextual information including:
  - *Scientific Name*
  - *Habitat*
  - *Damage Description*
  - *Control Methods & Prevention*

## Project Structure
```plaintext
pest-identification/
├── app.py                  # Main Flask backend application script
├── b.py                    # Standalone inference helper script (optional)
├── best.pt                 # Custom trained YOLO model weights
├── yolov8n.pt              # Fallback base YOLOv8 nano model
├── ip102.yaml              # Configuration file defining the 16 pest classes
├── requirements.txt        # Python dependencies list
├── static/                 # Folder for static assets (CSS, JS, output images)
│   └── results/            # Automatically handles saving bounding box images locally
└── templates/              # HTML files (contains index.html)
```

## Setup & Installation Instructions

This guide assumes you are running on Windows, but the steps are similar for Linux/Mac.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You can check your version by running:
```bash
python --version
```

### 2. Create a Virtual Environment (Recommended)
Isolating dependencies prevents conflicts with other local Python projects.
```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

### 3. Install Dependencies
Ensure you install the required packages (such as Flask, PyTorch, Ultralytics, OpenCV) from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Ensure the Model Exists
Make sure your custom trained model `best.pt` is inside the root of the `pest-identification` folder. The app detects it automatically. If it's missing, the app will attempt to use the `yolov8n.pt` base model instead.

### 5. Start the Application
Run the Flask server:
```bash
python app.py
```
By default, the application runs on `http://127.0.0.1:5000/`. Open this URL in your web browser, upload an image, and view your prediction results!
