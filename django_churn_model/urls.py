from django.contrib import admin
from django.urls import path
import django_churn_model_app.views as views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("training/", views.TrainChurnModelView.as_view(), name='model_training'),
    path("training/", views.PredChurnModelView.as_view(), name='model_prediction')
]
