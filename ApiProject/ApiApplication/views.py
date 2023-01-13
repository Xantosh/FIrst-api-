from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import *

# Create your views here.

class BookApiView(APIView):
    def get(self,request):
        allBooks = Book.objects.all().values()
        return Response({"Message":"List of Books", "Book List":allBooks})
    
    def post(self,request):
        Book.objects.create(id=request.data["id"],
                            title=request.data["title"],
                            author=request.data["author"])
        
        book = Book.objects.all().filter(id=request.data["id"]).values()
        return Response({"Message":"New Book Added", "Book List":book})