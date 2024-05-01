from django.urls import path

from MainPage import views
urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', fashion_recommender, name='upload'),

]