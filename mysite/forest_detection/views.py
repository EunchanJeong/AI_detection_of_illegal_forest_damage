from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'forest_detection/image_upload.html')
