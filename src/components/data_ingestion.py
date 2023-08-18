import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from sklearn.preprocessing import MinMaxScaler
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
            df_mat = pd.read_csv("notebook/data/student-por.csv", sep=";")
            
            # Feature Engineering
            df_mat = df_mat.drop(['famsup', 'Pstatus', 'paid', 'famsize'], axis=1)
            df_mat['total'] = df_mat['G1'] + df_mat['G2']
            df_mat['average_grade'] = df_mat['total'] / 2
            df_mat['Avg_Parent_Edu'] = (df_mat['Medu'] + df_mat['Fedu']) / 2
            df_mat['Parent_Job_Type'] = df_mat['Mjob'] + '_' + df_mat['Fjob']
            df_mat['Total_Alcohol'] = df_mat['Dalc'] + df_mat['Walc']
            df_mat['Grade_Improvement'] = df_mat['G2'] - df_mat['G1']
            df_mat['Study_Efficiency'] = df_mat['studytime'] / df_mat['freetime']
            days_in_school_year = 180  # Example value, modify as needed
            df_mat['Avg_Daily_Absences'] = df_mat['absences'] / days_in_school_year
            df_mat['School_Higher_Interaction'] = df_mat['school'] + '_' + df_mat['higher']
            age_bins = [15, 17, 20, 22]  # Example age groups, modify as needed
            age_labels = ['15-17', '18-20', '21-22']
            df_mat['Age_Group'] = pd.cut(df_mat['age'], bins=age_bins, labels=age_labels)
            df_mat['Internet_Studytime'] = df_mat['internet'] + '_' + df_mat['studytime'].astype(str)
            df_mat['Internet_Activities'] = df_mat['internet'] + '_' + df_mat['activities']
            scaler = MinMaxScaler()
            numeric_features = ['age', 'absences', 'G1', 'G2', 'total', 'average_grade', 'Avg_Parent_Edu', 'Total_Alcohol', 'Grade_Improvement', 'Study_Efficiency', 'Avg_Daily_Absences']
            df_mat[numeric_features] = scaler.fit_transform(df_mat[numeric_features])

            logging.info("Read dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df_mat.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df_mat, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Split dataset into train and test")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path))

