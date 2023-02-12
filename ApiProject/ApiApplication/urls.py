from django.urls import path
from ApiApplication import views  

urlpatterns = [
        path('',views.BookApiView.as_view()),
        path('state/' ,views.StateApiView.as_view()),
        path('train/',views.Train.as_view()),
        path('action/',views.ActionTransform.as_view())
        
        ]
