from django.urls import path
from .views import *

urlpatterns = [
    path("", predict_rainfall_view, name="predict_rainfall"),
    path("about", about, name="about"),
]
