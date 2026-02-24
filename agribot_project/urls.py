"""
URL configuration for agribot_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from core import views



urlpatterns = [
    path('admin_panel/', admin.site.urls), # Django's built-in admin
    path('', views.home, name='home'),
    path('prediction/', views.index, name='index'),
    path('register/', views.register, name='register'),
    
    # YOUR CUSTOM ADMIN
    path('admin-login/', views.admin_login, name='admin_login'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),

    path('admin-pending/', views.admin_pending, name='admin_pending'),
    path('approve/<int:user_id>/', views.approve_user, name='approve_user'),
    path('reject/<int:user_id>/', views.reject_user, name='reject_user'),

    path('admin-all-users/', views.admin_all_users, name='admin_all_users'),
    path('change-status/<int:user_id>/', views.change_status, name='change_status'),

    path('user-login/', views.user_login, name='user_login'),
    path('user-dashboard/', views.user_dashboard, name='user_dashboard'),

    path('user-myprofile/', views.my_profile, name='my_profile'),

    path('logout/', views.logout_view, name='logout'),

    path('user-editprofile/', views.edit_profile, name='edit_profile'),

    path('user-changepassword/', views.change_password, name='change_password'),

    path('user-cropdetails/', views.crop_details, name='crop_details'),
]

from django.conf import settings
from django.conf.urls.static import static

# Keep your static/media config below as it is
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



