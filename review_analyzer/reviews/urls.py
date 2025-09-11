from django.urls import path
from . import views

urlpatterns = [
    path('reviews/', views.review_list),
    path('analyze/', views.analyze_reviews),
]