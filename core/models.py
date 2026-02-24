from django.db import models

# Create your models here.
from django.contrib.auth.models import User

from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    # Change 'on_view' to 'on_delete'
    user = models.OneToOneField(User, on_delete=models.CASCADE) 
    fullname = models.CharField(max_length=200)
    contact = models.CharField(max_length=15)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)

    def __str__(self):
        return f'{self.user.username} Profile'