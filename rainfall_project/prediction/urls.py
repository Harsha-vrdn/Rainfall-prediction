from django.urls import path
from .views import predict_rainfall_view

urlpatterns = [
    path("", predict_rainfall_view, name="predict_rainfall"),
]
