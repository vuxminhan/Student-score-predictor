import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationCongig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join(os.path.join('artifacts', 'train.csv'))
    test_data_path: str=os.path.join(os.path.join('artifacts', 'test.csv'))
    raw_data_path: str=os.path.join(os.path.join('artifacts', 'raw_data.csv'))
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Enter data ingestion method or component")
        try:
            df_mat=pd.read_csv("notebook/data/student-mat.csv", sep = ";")
            logging.info("Read dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df_mat.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            train_set,test_set=train_test_split(df_mat,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Split dataset into train and test")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path))
