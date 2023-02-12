from django.urls import path
from policy import views  

urlpatterns = [
        path('',views.BookApiView.as_view()),
        path('train/',views.Train.as_view())
        
        ]
