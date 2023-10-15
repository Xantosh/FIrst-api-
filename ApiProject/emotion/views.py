from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .emotion import EmotionAnalyzer

class GetEmotion(APIView):
    def get(self,request):
        return Response({"Message":"Send a post request with sentence that you nedd emotion to "})
    
    def post(self,request):
        emotionAnalyzer = EmotionAnalyzer()
        sentence = request.data['sentence']
        x = emotionAnalyzer.GetEmotion(sentence)
        return Response({"emotion": x})
    





