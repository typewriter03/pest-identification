from flask import Flask, request, jsonify, render_template
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import time

app = Flask(__name__, static_folder='static')


# Load the model once when the server starts
model_path = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
model = YOLO(model_path)

# Load the class names from the YAML file (modify if required)
yaml_path = 'ip102.yaml'
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
custom_class_names = data['names']

# Set up the output directory
output_dir = 'static/results'
os.makedirs(output_dir, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # The HTML page for uploading images

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files['image']
    img_path = os.path.join(output_dir, file.filename)
    file.save(img_path)

    # Load and preprocess image
    img = cv2.imread(img_path)
    img0 = img.copy()

    # Get original image dimensions
    height, width = img.shape[:2]

    # Resize image for model inference
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))  # Resize to the model's expected input size
    img_resized = torch.from_numpy(img_resized).float() / 255.0  # Normalize the image to [0, 1]
    img_resized = img_resized.permute(2, 0, 1).unsqueeze(0)  # Change the image shape to (1, 3, 640, 640)

    # Define insect info (Add more as needed)
    insect_info = {
    "rice leaf roller": {
        "name": "Rice Leaf Roller",
        "description": "The larvae of this pest roll rice leaves to create a tubular shelter, feeding from within and protecting themselves from predators.",
        "scientific_name": "Cnaphalocrocis medinalis",
        "habitat": "Common in tropical and subtropical rice-growing regions, especially areas with consistent irrigation and high humidity.",
        "damage": "Rolled leaves reduce the plant's ability to perform photosynthesis, leading to stunted growth and delayed maturity.",
        "control_methods": "Use parasitoids like Trichogramma spp., light traps, and specific insecticides like cartap hydrochloride.",
        "prevention": "Timely planting and maintaining field hygiene."
    },
    "rice leaf caterpillar": {
        "name": "Rice Leaf Caterpillar",
        "description": "A polyphagous pest that feeds heavily on rice leaves, often during the night.",
        "scientific_name": "Spodoptera mauritia",
        "habitat": "Prefers areas with moist soil and abundant vegetation.",
        "damage": "Chews large sections of leaves, reducing photosynthesis, and can cut at the base, killing seedlings.",
        "control_methods": "Hand-picking larvae, using biocontrol agents like NPV, and applying neem oil sprays.",
        "prevention": "Avoid excessive irrigation and grow resistant rice varieties."
    },
    "paddy stem maggot": {
        "name": "Paddy Stem Maggot",
        "description": "Larvae bore into young rice stems, damaging vascular tissues.",
        "scientific_name": "Hydrellia philippina",
        "habitat": "Prefers shallow, water-logged rice paddies.",
        "damage": "Stunted growth due to interrupted nutrient flow and the appearance of 'silver shoots' that do not bear grain.",
        "control_methods": "Intermittent field drying, systemic insecticides like imidacloprid.",
        "prevention": "Early planting and crop rotation to disrupt pest lifecycle."
    },
    "asiatic rice borer": {
        "name": "Asiatic Rice Borer",
        "description": "A nocturnal moth whose larvae penetrate rice stems.",
        "scientific_name": "Chilo suppressalis",
        "habitat": "Rainfed and irrigated lowland rice fields.",
        "damage": "Causes 'dead hearts' in young plants and 'whiteheads' in mature plants, leading to yield losses up to 60%.",
        "control_methods": "Use pheromone traps, release egg parasitoids, and spray chlorantraniliprole.",
        "prevention": "Remove rice stubble after harvest to reduce overwintering populations."
    },
    "yellow rice borer": {
        "name": "Yellow Rice Borer",
        "description": "Larvae bore into stems, disrupting growth and leading to significant damage.",
        "scientific_name": "Scirpophaga incertulas",
        "habitat": "Irrigated and rainfed lowlands of Asia.",
        "damage": "Whitehead formation during the reproductive stage.",
        "control_methods": "Release biological agents like Tetrastichus spp., apply carbofuran granules.",
        "prevention": "Avoid excessive nitrogen fertilization and maintain crop diversity through intercropping."
    },
    "rice gall midge": {
        "name": "Rice Gall Midge",
        "description": "Induces gall formation ('onion shoots'), reducing productive tillers.",
        "scientific_name": "Orseolia oryzae",
        "habitat": "Waterlogged fields during monsoon seasons.",
        "damage": "Reduction in the number of productive tillers.",
        "control_methods": "Use resistant varieties like Tadukan, apply granular insecticides like carbofuran.",
        "prevention": "Avoid staggered planting."
    },
    "rice stemfly": {
        "name": "Rice Stemfly",
        "description": "Larvae burrow into stems, disrupting nutrient flow.",
        "scientific_name": "Melanagromyza sp.",
        "habitat": "Moist lowland areas.",
        "damage": "Reduced tillering significantly affects plant productivity.",
        "control_methods": "Similar to rice gall midge controls.",
        "prevention": "Crop rotation and removal of infested plants."
    },
    "brown plant hopper": {
        "name": "Brown Plant Hopper",
        "description": "A sap-sucking pest that transmits viral diseases in rice crops.",
        "scientific_name": "Nilaparvata lugens",
        "habitat": "Warm and humid rice fields.",
        "damage": "Causes 'hopper burn' and transmits tungro virus.",
        "control_methods": "Apply systemic insecticides and grow resistant varieties.",
        "prevention": "Maintain optimal plant density and avoid over-fertilization."
    },
    "white backed plant hopper": {
        "name": "White Backed Plant Hopper",
        "description": "A sap-feeding pest that damages rice crops and spreads diseases.",
        "scientific_name": "Sogatella furcifera",
        "habitat": "Lowland rice fields.",
        "damage": "Similar to brown plant hopper damage, including hopper burn.",
        "control_methods": "Use resistant varieties and targeted insecticides.",
        "prevention": "Avoid excessive nitrogen fertilization."
    },
    "small brown plant hopper": {
        "name": "Small Brown Plant Hopper",
        "description": "Another hopper species that damages rice crops through sap-sucking.",
        "scientific_name": "Laodelphax striatellus",
        "habitat": "Rice paddies with standing water.",
        "damage": "Stunts growth and reduces grain yield.",
        "control_methods": "Spray systemic insecticides.",
        "prevention": "Proper drainage and field hygiene."
    },
    "rice water weevil": {
        "name": "Rice Water Weevil",
        "description": "Adults feed on leaves, while larvae damage roots.",
        "scientific_name": "Lissorhoptrus oryzophilus",
        "habitat": "Flooded rice fields.",
        "damage": "Root feeding by larvae reduces nutrient uptake, stunting growth.",
        "control_methods": "Apply insecticides like pyrethroids and drain fields periodically.",
        "prevention": "Field draining before planting reduces larval survival."
    },
    "rice leafhopper": {
        "name": "Rice Leafhopper",
        "description": "A vector of rice tungro virus that feeds on rice plant sap.",
        "scientific_name": "Nephotettix virescens",
        "habitat": "Rice fields with young seedlings.",
        "damage": "Yellowing and stunting of rice plants.",
        "control_methods": "Grow resistant varieties and apply targeted insecticides.",
        "prevention": "Monitor populations using yellow sticky traps."
    },
    "grain spreader thrips": {
        "name": "Grain Spreader Thrips",
        "description": "Damages rice grains by feeding on developing kernels.",
        "scientific_name": "Stenchaetothrips biformis",
        "habitat": "Rice fields with high humidity.",
        "damage": "Discolored and malformed grains.",
        "control_methods": "Apply insecticidal sprays at flowering stage.",
        "prevention": "Field hygiene and early planting."
    },
    "rice shell pest": {
        "name": "Rice Shell Pest",
        "description": "A pest that damages rice grains by feeding on the hull.",
        "scientific_name": "Parapoynx stagnalis",
        "habitat": "Waterlogged fields.",
        "damage": "Reduces grain quality and yield.",
        "control_methods": "Use light traps to monitor adults and apply insecticides.",
        "prevention": "Crop rotation and avoiding over-flooding."
    },
    "grub": {
        "name": "Grub",
        "description": "Larvae of beetles that damage rice roots.",
        "scientific_name": "Phyllophaga spp.",
        "habitat": "Soil in rice fields.",
        "damage": "Roots are chewed, causing plant wilting.",
        "control_methods": "Use soil-applied insecticides and encourage natural predators like birds.",
        "prevention": "Plowing fields to expose larvae to predators."
    },
    "mole cricket": {
        "name": "Mole Cricket",
        "description": "Burrows into soil, cutting rice roots and stems.",
        "scientific_name": "Gryllotalpa spp.",
        "habitat": "Moist soils in lowland fields.",
        "damage": "Plant wilting and reduced tillering.",
        "control_methods": "Flood fields to reduce habitat suitability and apply baits.",
        "prevention": "Maintain proper drainage."
    }
}


    # Run inference
    results = model(img_resized)  # Inference on the image
    output_image = img0.copy()

    # Process detections
    detections = []
    extra_info = {}
    for result in results[0].boxes.data:  # Access the first result and iterate through boxes
        xyxy = result[:4].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        conf = result[4].item()  # Confidence score
        cls = int(result[5].item())  # Class index

        if 0 <= cls < len(custom_class_names):
            label = f'{custom_class_names[cls]} {conf:.2f}'  # Get label and confidence score
            detections.append(label)

            # Add additional information for the insect class (only for the detected insect)
            insect_class = custom_class_names[cls]
            if insect_class in insect_info:
                extra_info = insect_info[insect_class]  # Store the info of the detected insect

            # Rescale bounding box coordinates back to the original image size
            aspect_ratio = min(640 / width, 640 / height)
            dw = (640 - aspect_ratio * width) / 2
            dh = (640 - aspect_ratio * height) / 2

            # Scale the bounding box
            x1, y1, x2, y2 = xyxy
            x1 = (x1 - dw) / aspect_ratio
            y1 = (y1 - dh) / aspect_ratio
            x2 = (x2 - dw) / aspect_ratio
            y2 = (y2 - dh) / aspect_ratio

            # Draw bounding box and label on the original image
            xyxy = [int(i) for i in [x1, y1, x2, y2]]
            cv2.rectangle(output_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(output_image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image with a dynamic filename
    timestamp = int(time.time())  # Use timestamp to make filename unique
    output_image_path = os.path.join(output_dir, f"output_image_{timestamp}.jpg")
    cv2.imwrite(output_image_path, output_image)

    return jsonify({
        'detections': detections,
        'extra_info': extra_info,  # Send only the info of the detected insect
        'image_url': f'/static/results/output_image_{timestamp}.jpg'  # Use dynamic filename
    })



if __name__ == '__main__':
    app.run(debug=True)
