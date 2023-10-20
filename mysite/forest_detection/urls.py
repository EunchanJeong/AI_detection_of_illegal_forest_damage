from django.urls import path

from . import views

app_name = 'forest_detection'

urlpatterns = [
    path('', views.index, name='forest_detection'),
]