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
from django.contrib.auth.models import User

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
crop_model_path = os.path.join(BASE_DIR, 'core', 'models', 'crop_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'core', 'models', 'label_encoder.pkl')
disease_model_path = os.path.join(BASE_DIR, 'core', 'models', 'disease_model.pth')

crop_model = joblib.load(crop_model_path)
label_encoder = joblib.load(encoder_path)

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

def index(request):
    """View for the AI Prediction tools"""
    context = {}
    if request.method == 'POST':
        # Handle Crop Recommendation
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

        # Handle Disease Prediction
        elif 'image' in request.FILES:
            try:
                img = Image.open(request.FILES['image']).convert('RGB')
                img_t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = disease_model(img_t)
                    _, predicted = torch.max(output, 1)
                    idx = predicted.item()
                    context['disease_result'] = CLASS_NAMES[idx].replace('___', ' ').replace('_', ' ')
            except Exception as e:
                context['disease_result'] = f"Error: {str(e)}"

    return render(request, 'index.html', context)


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

def reject_user(request, user_id):
    user = User.objects.get(id=user_id)
    user.delete() # Deleting the user also deletes their profile due to CASCADE
    return redirect('admin_pending')


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

def user_dashboard(request):
    # This renders the dashboard shown in your last image
    return render(request, 'user_dashboard.html')


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