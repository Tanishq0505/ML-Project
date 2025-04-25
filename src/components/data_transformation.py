import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer # it is used to apply different transformations to different columns of the dataframe 
from sklearn.impute import SimpleImputer # it is used to fill the missing values in the dataframe 
from sklearn.pipeline import Pipeline # it is used to create a pipeline of different transformations to be applied to the dataframe
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object # it is used to save the object in the specified path

@dataclass # 
class DataTransformationConfig: # it is used to create a class with the specified attributes and default values 
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
# it is used to create a file path for the preprocessor object to be saved in the artifacts folder
class DataTransformation:# it is used to create a class for data transformation
    def __init__(self): # it is used to create a constructor for the class
        self.data_transformation_config=DataTransformationConfig() # it is used to create an object of the DataTransformationConfig class

    def get_data_transformer_object(self):
        '''
        This function is  responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline( # it is used to create a pipeline of different transformations to be applied to the numerical columns of the dataframe
                steps=[
                ("imputer",SimpleImputer(strategy="median")), # it is used to fill the missing values in the numerical columns of the dataframe with the median value of the column
                ("scaler",StandardScaler()) # it is used to scale the numerical columns of the dataframe to have a mean of 0 and a standard deviation of 1

                ]
            )

            cat_pipeline=Pipeline( # it is used to create a pipeline of different transformations to be applied to the categorical columns of the dataframe

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), # it is used to fill the missing values in the categorical columns of the dataframe with the most frequent value of the column
                ("one_hot_encoder",OneHotEncoder()), # it is used to convert the categorical columns of the dataframe into numerical columns by creating dummy variables for each category in the column
                ("scaler",StandardScaler(with_mean=False)) # it is used to scale the categorical columns of the dataframe to have a mean of 0 and a standard deviation of 1
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}") # it is used to log the categorical columns of the dataframe
            logging.info(f"Numerical columns: {numerical_columns}") # it is used to log the numerical columns of the dataframe

            preprocessor=ColumnTransformer( # it is used to apply different transformations to different columns of the dataframe
                [
                ("num_pipeline",num_pipeline,numerical_columns),# it is used to apply the numerical pipeline to the numerical columns of the dataframe
                ("cat_pipelines",cat_pipeline,categorical_columns) # it is used to apply the categorical pipeline to the categorical columns of the dataframe

                ]


            )

            return preprocessor # it is used to return the preprocessor object created by the ColumnTransformer class
        
        except Exception as e:
            raise CustomException(e,sys) # it is used to raise an exception if there is any error in the code
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() # it is used to get the preprocessor object created by the get_data_transformer_object function

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) # it is used to drop the target column from the training dataframe
            # axis=1 is used to drop the column from the dataframe
            target_feature_train_df=train_df[target_column_name] # it is used to get the target column from the training dataframe

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) # it is used to fit the preprocessor object on the training dataframe and transform the training dataframe
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)# it is used to transform the testing dataframe using the preprocessor object fitted on the training dataframe

            train_arr = np.c_[   # it is used to concatenate the input features and target features of the training dataframe into a single array
                # np.c_ is used to concatenate the two arrays column-wise 
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # it is used to concatenate the input features and target features of the testing dataframe into a single array

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path, # it is used to save the preprocessor object in the specified path
                # file_path is the path where the object is to be saved
                obj=preprocessing_obj  # it is used to save the preprocessor object created by the ColumnTransformer class

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path, # it is used to return the path where the preprocessor object is saved
            )
        except Exception as e:
            raise CustomException(e,sys)