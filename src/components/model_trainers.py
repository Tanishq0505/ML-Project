import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models # it is used to save the object in the specified path and evaluate the models

@dataclass
class ModelTrainerConfig: # it is used to create a class with the specified attributes and default values
    trained_model_file_path=os.path.join("artifacts","model.pkl") # it is used to create a file path for the trained model to be saved in the artifacts folder

class ModelTrainer: # it is used to create a class for model training
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig() #  it is used to create an object of the ModelTrainerConfig class


    def initiate_model_trainer(self,train_array,test_array): # it is used to create a method for model training
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=( # it is used to split the training and testing data into input and output variables
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params) # it is used to evaluate the models using the training and testing data and return the model report
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6: # it is used to check if the best model score is less than 0.6
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object( # it is used to save the best model in the specified path
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted=best_model.predict(X_test) # it is used to predict the output using the best model on the testing data

            r2_square = r2_score(y_test, predicted) # it is used to calculate the r2 score of the predicted output and the actual output
            return r2_square # it is used to return the r2 score of the predicted output and the actual output
            



            
        except Exception as e:
            raise CustomException(e,sys)