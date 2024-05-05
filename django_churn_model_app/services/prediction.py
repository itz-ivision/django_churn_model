import os
import pandas as pd
import numpy as np

from rest_framework import status
import pickle
import json


base_path = os.getcwd()
pikle_path = os.path.normpath(base_path+os.sep+'pickle')
log_path = os.path.normpath(base_path+os.sep+'log')


class Prediction:

    def predict(self,request):

        return_dict = dict()

        try: 
            input_request = request.body
            decode_input_request = input_request.decode('utf-8').replace("'",'"')
            request_dict = json.loads(decode_input_request)
            df_pred = pd.json_normalize(request_dict)

            pickle_file = os.path.normpath(base_path+os.sep+'model.sav')
            model = pickle.load(pickle_file, 'rb')
            prediction = model.predict(df_pred)

            request_dict['prediction'] = prediction
            return_dict['response'] = return_dict
            return_dict['status'] = status.HTTP_200_OK
            return request_dict
        
        except Exception as e:
            return_dict['response'] = "Exception when predicting  :  " + str(e.__str__)
            return_dict['status'] = status.HTTP_500_INTERNAL_SERVER_ERROR
            return request_dict