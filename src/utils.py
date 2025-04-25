import os
import sys

import numpy as np 
import pandas as pd
import dill # it is used to serialize and deserialize the python objects 
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj): # it is used to save the object in the specified path
    try:
        dir_path = os.path.dirname(file_path) # it is used to get the directory path of the file

        os.makedirs(dir_path, exist_ok=True) # it is used to create the directory if it does not exist

        with open(file_path, "wb") as file_obj: # it is used to open the file in write binary mode
            pickle.dump(obj, file_obj) # it is used to dump the object in the file

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))): # it is used to iterate through the models
            model = list(models.values())[i] # it is used to get the model from the list of models
            para=param[list(models.keys())[i]] # it is used to get the parameters of the model from the list of parameters

            gs = GridSearchCV(model,para,cv=3) # it is used to perform grid search cross validation on the model with the parameters
            gs.fit(X_train,y_train) # it is used to fit the model on the training data

            model.set_params(**gs.best_params_) # it is used to set the best parameters of the model
            model.fit(X_train,y_train) # it is used to fit the model on the training data


            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score # it is used to store the model name and the score of the model in the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path): # it is used to load the object from the specified path
    try:
        with open(file_path, "rb") as file_obj: # it is used to open the file in read binary mode
            return pickle.load(file_obj) # it is used to load the object from the file

    except Exception as e:
        raise CustomException(e, sys)