from django.shortcuts import render, redirect
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import os
from django.contrib.auth.models import User
from .models import Profile
import pickle
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .models import ChatLog

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
# Path joining is safer for cloud environments
# --- REPLACE YOUR MODEL LOADING SECTION WITH THIS ---

# Use settings.BASE_DIR to point to /workspaces/AgriBot
MODEL_DIR = os.path.join(settings.BASE_DIR, 'core', 'models')

crop_model_path = os.path.join(MODEL_DIR, 'crop_model.pkl')
encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
disease_model_path = os.path.join(MODEL_DIR, 'disease_model.pth')

# Load the new 4-feature synced files
ferti_model = joblib.load(os.path.join(MODEL_DIR, 'fertilizer_model.pkl'))
ferti_crop_enc = joblib.load(os.path.join(MODEL_DIR, 'ferti_crop_encoder.pkl'))
ferti_label_enc = joblib.load(os.path.join(MODEL_DIR, 'ferti_label_encoder.pkl'))

# Initialize as None so the server doesn't crash if loading fails
crop_model = None
label_encoder = None

try:
    if os.path.exists(crop_model_path):
        crop_model = joblib.load(crop_model_path)
        print("✅ Crop Model loaded successfully.")
    
    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
        print("✅ Label Encoder loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}") 
    # This allows the server to keep running so you can debug

disease_model = models.resnet18()
disease_model.fc = nn.Linear(disease_model.fc.in_features, 38)
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# 3. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- VIEW FUNCTIONS ---

def home(request):
    """View for the professional landing page"""
    return render(request, 'home.html')


def register(request):
    context = {}
    if request.method == 'POST':
        u_name = request.POST.get('username')
        f_name = request.POST.get('fullname')
        email = request.POST.get('email')
        pwd = request.POST.get('password')
        cont = request.POST.get('contact')
        cty = request.POST.get('city')
        st = request.POST.get('state')
        pic = request.FILES.get('picture')

        try:
            # Create user and set to INACTIVE (Pending)
            user = User.objects.create_user(username=u_name, email=email, password=pwd)
            user.is_active = False  # This ensures they go to Pending list
            user.save()
            
            Profile.objects.create(user=user, fullname=f_name, contact=cont, city=cty, state=st, picture=pic)
            
            context['success_msg'] = True
            return render(request, 'register.html', context)
            
        except Exception as e:
            # This will help you see the EXACT error in your terminal
            print(f"DEBUG ERROR: {e}") 
            # This sends the error message back to the UI
            context['error_msg'] = str(e)

    return render(request, 'register.html', context)


def admin_login(request):
    context = {}
    if request.method == 'POST':
        user = request.POST.get('admin_user')
        pas = request.POST.get('admin_pass')

        # Use the fixed credentials you requested
        if user == "admin" and pas == "agribot123":
            context['login_success'] = True
            # We return render instead of redirect so the JS modal can show first
            return render(request, 'admin_login.html', context)
        else:
            context['error'] = True
            
    return render(request, 'admin_login.html', context)

def admin_dashboard(request):
    # Fetching counts for the cards in your screenshot
    total_users = Profile.objects.count()
    # Assuming 'is_active=False' means pending
    pending_users = User.objects.filter(is_active=False).count() 
    
    context = {
        'total_count': total_users,
        'pending_count': pending_users,
        'restricted_count': 0  # Placeholder for restricted users
    }
    return render(request, 'admin_dashboard.html', context)


def admin_pending(request):
    # Fetching users who are NOT active (is_active=False)
    # We select_related('profile') to get their pictures and full names in one query
    pending_users = Profile.objects.filter(user__is_active=False)
    return render(request, 'admin_pending.html', {'pending_users': pending_users})

def approve_user(request, user_id):
    user = User.objects.get(id=user_id)
    user.is_active = True # Move from Pending to All Users
    user.save()
    
    # Reload pending users and send success flag
    pending_users = Profile.objects.filter(user__is_active=False)
    return render(request, 'admin_pending.html', {
        'pending_users': pending_users, 
        'approve_success': True # Triggers the modal
    })

def admin_all_users(request):
    # Fetch all users who are already approved (is_active=True)
    # We exclude the admin themselves from this list
    active_profiles = Profile.objects.filter(user__is_active=True).exclude(user__is_staff=True)
    return render(request, 'admin_all_users.html', {'active_profiles': active_profiles})

def change_status(request, user_id):
    # This toggles the user from active to inactive
    user = User.objects.get(id=user_id)
    user.is_active = False 
    user.save()
    return redirect('admin_all_users')

def reject_user(request, user_id):
    try:
        user = User.objects.get(id=user_id)
        user.delete() # Removes both User and Profile via CASCADE
        
        # Reload pending list and send rejection flag
        pending_users = Profile.objects.filter(user__is_active=False)
        return render(request, 'admin_pending.html', {
            'pending_users': pending_users, 
            'reject_success': True 
        })
    except User.DoesNotExist:
        return redirect('admin_pending')


from django.contrib.auth import authenticate, login

def user_login(request):
    if request.method == 'POST':
        e = request.POST.get('email')
        p = request.POST.get('password')

        # Since Django authenticates by username, we find the username associated with the email
        try:
            user_obj = User.objects.get(email=e)
            user = authenticate(username=user_obj.username, password=p)
            
            if user is not None:
                login(request, user)
                # Pass success flag to show modal before redirecting
                return render(request, 'user_login.html', {'login_success': True})
            else:
                return render(request, 'user_login.html', {'error': "Invalid Credentials or Account Pending Approval"})
        except User.DoesNotExist:
            return render(request, 'user_login.html', {'error': "Email not found"})

    return render(request, 'user_login.html')


from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User # Built-in model
# REMOVE the UserProfile import line that caused the error

@login_required
def user_dashboard(request):
    total_users_count = User.objects.count()
    active_logins_count = User.objects.filter(last_login__isnull=False).count()
    
    # FETCH REAL COUNT FROM CHATLOG
    bot_interactions_count = ChatLog.objects.count() 

    context = {
        'total_users': total_users_count,
        'user_logins': active_logins_count,
        'bot_interactions': bot_interactions_count, # Now real-time!
    }
    return render(request, 'user_dashboard.html', context)


from django.contrib.auth.decorators import login_required

@login_required
def my_profile(request):
    # Fetch the profile linked to the logged-in user
    user_profile = Profile.objects.get(user=request.user)
    return render(request, 'my_profile.html', {'profile': user_profile})

from django.contrib.auth import logout
from django.shortcuts import redirect

def logout_view(request):
    logout(request)
    return redirect('home') # Redirects back to the landing page after logout

from django.contrib import messages

@login_required
def edit_profile(request):
    profile = Profile.objects.get(user=request.user)
    
    if request.method == 'POST':
        # Get data from the form
        profile.fullname = request.POST.get('fullname')
        profile.contact = request.POST.get('contact')
        profile.city = request.POST.get('city')
        profile.state = request.POST.get('state')
        
        # Check if a new picture was uploaded
        if request.FILES.get('picture'):
            profile.picture = request.FILES.get('picture')
            
        profile.save()
        # Trigger success flag for modal
        return render(request, 'edit_profile.html', {'profile': profile, 'update_success': True})

    return render(request, 'edit_profile.html', {'profile': profile})


from django.contrib.auth import update_session_auth_hash

@login_required
def change_password(request):
    if request.method == 'POST':
        old_p = request.POST.get('old_password')
        new_p = request.POST.get('new_password')
        confirm_p = request.POST.get('confirm_password')

        # 1. Verify old password
        if not request.user.check_password(old_p):
            return render(request, 'change_password.html', {'error': "Old password is incorrect"})

        # 2. Check if new passwords match
        if new_p != confirm_p:
            return render(request, 'change_password.html', {'error': "New passwords do not match"})

        # 3. Update password
        request.user.set_password(new_p)
        request.user.save()
        
        # 4. Keep user logged in
        update_session_auth_hash(request, request.user)
        return render(request, 'change_password.html', {'pass_success': True})

    return render(request, 'change_password.html')


from django.contrib.auth.decorators import login_required

@login_required
def crop_details(request):
    return render(request, 'crop_details.html')

@login_required
def fertilizer_details(request):
    return render(request, 'user_fertilizer.html')


import time
from django.http import JsonResponse


from django.shortcuts import render, redirect

@login_required
def chatbot_init(request):
    # Check if the model is already initialized for this session
    if request.session.get('chatbot_initialized', False):
        return redirect('chatbot_main')
    return render(request, 'user_chatbot_init.html')

@login_required
def initialize_model_api(request):
    # This is called by your AJAX/Fetch request during animation
    import time
    time.sleep(3) # Simulating loading weights/NLP logic
    
    # SET SESSION STATE: Mark as initialized
    request.session['chatbot_initialized'] = True
    return JsonResponse({'status': 'ready'})

@login_required
def chatbot_main(request):
    # Optional: Safety check to ensure they don't bypass the loader
    if not request.session.get('chatbot_initialized', False):
        return redirect('chatbot_init')
    return render(request, 'user_chatbot_main.html')


@login_required
def chatbot_page(request):
    return render(request, 'user_chatbot_page.html')


@login_required
def crop_recommendation(request):
    if request.method == 'POST':
        try:
            # Using long names for clarity and professional structure
            n = float(request.POST.get('nitrogen'))
            p = float(request.POST.get('phosphorus'))
            k = float(request.POST.get('potassium'))
            temp = float(request.POST.get('temperature'))
            hum = float(request.POST.get('humidity'))
            ph = float(request.POST.get('ph'))
            rain = float(request.POST.get('rainfall'))
            
            # Features must be in this exact order for the XGBoost model
            inputs = np.array([[n, p, k, temp, hum, ph, rain]])
            
            if crop_model and label_encoder:
                prediction = crop_model.predict(inputs)
                # Convert numeric index back to crop name
                final_crop = label_encoder.inverse_transform(prediction)[0]
                
                request.session['crop_result'] = final_crop
                return redirect('crop_predict_result')
            else:
                return render(request, 'user_crop_recommend.html', {'error': "Model files missing in core/models/"})
            
        except Exception as e:
            return render(request, 'user_crop_recommend.html', {'error': f"Prediction Error: {str(e)}"})
            
    return render(request, 'user_crop_recommend.html')

@login_required
def crop_predict_result(request):
    # Retrieve result and display on the results page
    result = request.session.get('crop_result', 'Unknown')
    return render(request, 'crop_predict.html', {'result': result})


# core/views.py
from .disease_info import DISEASE_DETAILS # We will create this file next

@login_required
def disease_prediction(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            img = Image.open(image_file).convert('RGB')
            
            # Preprocess and Run ResNet-18
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = disease_model(img_t)
                _, predicted = torch.max(output, 1)
                idx = predicted.item()
            
            # Get the formatted class name (e.g., 'Corn - Grey Leaf Spot')
            raw_class = CLASS_NAMES[idx]
            display_name = raw_class.replace('___', ' - ').replace('_', ' ')
            
            # Fetch details from our data dictionary
            # If a class isn't in the dict yet, it provides a default message
            report_data = DISEASE_DETAILS.get(display_name, {
                'crop': display_name.split(' - ')[0],
                'disease': display_name.split(' - ')[1] if ' - ' in display_name else 'Healthy',
                'cause': "Detailed cause information for this specific plant disease is being updated in our database.",
                'prevention': ["Maintain proper irrigation.", "Ensure adequate sunlight.", "Consult a local agricultural expert."]
            })
            
            return render(request, 'disease_predict_result.html', {'details': report_data})
            
        except Exception as e:
            return render(request, 'user_disease_predict.html', {'error': f"Error: {str(e)}"})

    return render(request, 'user_disease_predict.html')


from .fertilizer_info import FERTILIZER_DATA

@login_required
def fertilizer_recommendation(request):
    """
    Handles Fertilizer Prediction. 
    GET: Shows the input form (fertilizer.html)
    POST: Processes data and shows report (fertilizer_predict.html)
    """
    if request.method == 'POST':
        try:
            # 1. Capture inputs from the form
            n = float(request.POST.get('nitrogen'))
            p = float(request.POST.get('phosphorous'))
            k = float(request.POST.get('potassium'))
            crop_name = request.POST.get('crop_type', '').strip().lower()

            # 2. Transform crop name to numeric index
            # Uses the encoder trained on your 300-row dataset
            crop_encoded = ferti_crop_enc.transform([crop_name])[0]

            # 3. Predict using the 4-feature XGBoost model
            # Order: Nitrogen, Phosphorous, Potassium, Crop Type
            input_data = [[n, p, k, crop_encoded]]
            prediction_idx = ferti_model.predict(input_data)[0]
            
            # 4. Convert numeric prediction to Fertilizer Name
            ferti_name = ferti_label_enc.inverse_transform([prediction_idx])[0]

            # 5. Fetch detailed "Suggestions" from fertilizer_info.py
            info = FERTILIZER_DATA.get(ferti_name.strip(), {
                'message': f"Recommended Fertilizer: {ferti_name}",
                'suggestions': ["No specific automated suggestions available for this type."]
            })

            # 6. RENDER THE RESULT PAGE (fertilizer_predict.html)
            return render(request, 'fertilizer_predict.html', {
                'message': info['message'],
                'suggestions': info['suggestions']
            })

        except Exception as e:
            # If an error occurs (like unseen label), stay on the form page
            return render(request, 'fertilizer.html', {'error': f"Prediction Error: {str(e)}"})

    # Default GET request: Show the input form
    return render(request, 'fertilizer.html')



from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import json
import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load the trained files
model = load_model('core/models/chatbot_model.h5')
intents = json.loads(open('core/intents.json').read())
words = pickle.load(open('core/models/words.pkl', 'rb'))
classes = pickle.load(open('core/models/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Add ChatLog to your imports at the top


@login_required
def chatbot_view(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        ints = predict_class(message)
        res = get_response(ints, intents)
        
        # SAVE THE INTERACTION TO THE DATABASE
        ChatLog.objects.create(user=request.user, query=message, response=res)
        
        return JsonResponse({'response': res})
    return render(request, 'chatbot_popup.html')