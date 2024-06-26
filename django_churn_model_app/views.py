from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from django_churn_model_app.services.training import Training
from django_churn_model_app.services.prediction import Prediction


class TrainChurnModelView(APIView):

    def get(self, request):

        train_obj = Training()
        response_dict = train_obj.train(request= request)
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status=status_value)
    

class PredChurnModelView(APIView):

    def post(self, request):
        pred_obj = Prediction()
        response_dict = pred_obj.predict(request= request)
        response = response_dict['response']
        status_value = response_dict['status']
        return Response(response, status=status_value)
