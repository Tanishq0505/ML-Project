import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # it is used to create classes that are mainly used to store data and have little functionality.

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainers import ModelTrainerConfig
from src.components.model_trainers import ModelTrainer
@dataclass
class DataIngestionConfig: 
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:   # This class is responsible for data ingestion, which includes reading the data from a CSV 
    # file, splitting it into training and testing sets, and saving these sets to specified paths.

    def __init__(self): # This is the constructor method that initializes the DataIngestion class.
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self): # This method is responsible for initiating the data ingestion process.
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe') #

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # This line creates 
            # the directory for the train data path if it does not exist.
        

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # This line saves the raw data to a CSV file at the specified path.

            logging.info("Train test split initiated") # This line logs that the train-test split process has started.
            # The train_test_split function is used to split the data into training and testing sets.
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)


            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
# This line saves the training set to a CSV file at the specified path.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
# This line saves the testing set to a CSV file at the specified path.

            logging.info("Ingestion of the data iss completed")

            return( 
                self.ingestion_config.train_data_path, # This line returns the paths of the training and testing sets.
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion() # This line creates an instance of the DataIngestion class.
    train_data,test_data=obj.initiate_data_ingestion() # This line calls the initiate_data_ingestion method to 
    # start the data ingestion process.

    data_transformation=DataTransformation() # This line creates an instance of the DataTransformation class.
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) # This line calls 
    # the initiate_data_transformation method to start the data transformation process.

    modeltrainer=ModelTrainer() # This line creates an instance of the ModelTrainer class.
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    # This line calls the initiate_model_trainer method to start the model training process.
    # It prints the report of the model training process.


