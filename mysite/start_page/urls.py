from django.urls import path

from . import views

app_name = 'start_page'

urlpatterns = [
    path('', views.index, name='start_page'),
]