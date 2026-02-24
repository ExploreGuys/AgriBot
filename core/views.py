from django.shortcuts import render
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import os

# 1. Global Variables & Class Names
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# 2. Load Models Once (at startup)
crop_model = joblib.load('core/models/crop_model.pkl')
label_encoder = joblib.load('core/models/label_encoder.pkl')

disease_model = models.resnet18()
num_classes = 38 
disease_model.fc = nn.Linear(disease_model.fc.in_features, num_classes)
disease_model.load_state_dict(torch.load('core/models/disease_model.pth', map_location=torch.device('cpu')))
disease_model.eval()

# 3. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Main Index View
def index(request):
    context = {}
    if request.method == 'POST':
        # --- Handle Crop Recommendation ---
        if 'n' in request.POST:
            try:
                inputs = np.array([[
                    float(request.POST.get('n')), float(request.POST.get('p')),
                    float(request.POST.get('k')), float(request.POST.get('temp')),
                    float(request.POST.get('hum')), float(request.POST.get('ph')),
                    float(request.POST.get('rain'))
                ]])
                prediction = crop_model.predict(inputs)
                context['crop_result'] = label_encoder.inverse_transform(prediction)[0]
            except Exception as e:
                context['crop_result'] = f"Error: {str(e)}"

        # --- Handle Disease Prediction ---
        elif 'image' in request.FILES:
            try:
                img = Image.open(request.FILES['image']).convert('RGB')
                img_t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = disease_model(img_t)
                    _, predicted = torch.max(output, 1)
                    idx = predicted.item()
                    # Convert internal name to user-friendly name
                    context['disease_result'] = CLASS_NAMES[idx].replace('___', ' ').replace('_', ' ')
            except Exception as e:
                context['disease_result'] = f"Error processing image: {str(e)}"

    return render(request, 'index.html', context)