from django.urls import path
from emotion import views  

urlpatterns = [
        path('',views.GetEmotion.as_view()),        
        ]
