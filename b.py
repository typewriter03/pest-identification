import torch

# Set the local path to the YOLOv5 directory and model
model_path = 'Mini/Test1/runs/detect/train2/weights/best.pt'
yolov5_repo_path = 'Mini/frontend/yolov5'  # Local path to your YOLOv5 repo

# Add the YOLOv5 directory to system path
import sys
sys.path.insert(0, yolov5_repo_path)

# Import the custom model class
from models.common import DetectMultiBackend

# Load the model
model = DetectMultiBackend(model_path)

# Print the class names
print(model.names)  # This should print the list of class names the model was trained with
