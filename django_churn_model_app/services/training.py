import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from rest_framework import status
from urllib.parse import urlparse
import constant as AppConst
import pickle


base_path = os.getcwd() 
data_path=os.path.normpath(base_path+os.sep+'data')
pickle_path=os.path.normpath(base_path+os.sep+'pickle')
log_path=os.path.normpath(base_path+os.sep+'log')

class Training:

    def train(self,request):

        return_dict = dict()

        try:
            train_data = os.normpath(data_path, os.sep, 'train_data.csv')
            df = pd.read_csv(train_data)
            df = df.fillna(0)
            X,y = self.get_feat_and_target(df, AppConst.TARGET)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestClassifier(max_depth=AppConst.MAX_DEPTH, n_estimators=AppConst.N_ESTIMATORS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy,precision,recall, f1score = self.accuracymeasures(y_test, y_pred)

            pickle_file = os.path.normpath(pickle_path+os.sep+'model.sav')
            pickle.dump(model, open(pickle_file, 'wb'))

            return_dict['response'] = "Model Trained Successfully."
            return_dict['status'] = status.HTTP_200_OK

            return return_dict
        except Exception as e:
            return_dict['response']="Exception when training the model: "+str(e.__str__)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict 
        
    def accuracymeasures(self, y_test, y_pred, avg_method):

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg_method)
        recall = recall_score(y_test, y_pred, average=avg_method)
        f1score = f1_score(y_test, y_pred, average=avg_method)
        target_report = ['0', '1']

        print("Classification Report ")
        print("--------------------------------",'\n')
        print(classification_report(y_test, y_pred, target_names=target_report))
        print("--------------------------------")
        print("Confusion Matrix ")
        print("--------------------------------", '\n')
        print(confusion_matrix(y_test, y_pred), '\n')

        print("Accuracy Mesures ")
        print("--------------------------------", '\n')
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1score)

        return accuracy,precision,recall, f1score

    def get_feat_and_target(self, df, target):
        """
            This function extracts the features (X) and target variable (y) from the given DataFrame.

            Parameters:
            df (pandas.DataFrame): The input DataFrame containing the data.
            target (str): The name of the target variable column in the DataFrame.

            Returns:
            X (pandas.DataFrame): A DataFrame containing the features (excluding the target variable).
            y (pandas.Series): A Series containing the target variable values.

            Raises:
            ValueError: If the target variable is not found in the DataFrame.
        """
        try:
            X = df.drop(target, axis=1)
            y = df[target]
        except KeyError:
            raise ValueError(f"The target variable '{target}' was not found in the DataFrame.")

        return X, y