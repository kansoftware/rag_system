from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('history/', views.history_list_view, name='history_list'),
    path('history/<int:pk>/', views.history_detail_view, name='history_detail'),
    path('history/<int:pk>/delete/', views.history_delete_view, name='history_delete'),
]