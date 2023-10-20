from django.urls import path

from . import views

app_name = 'satellite_map'

urlpatterns = [
    path('', views.index, name='satellite_map'),
]