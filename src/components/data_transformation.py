import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException   
from src.logger import logging
from src.utils import save_object

import os

@dataclass
class DataTransformationCongig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor_obj.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationCongig()
    def get_data_transformation_object(self):
        try:
            numerical_col = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
            categorical_col = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        
            num_pipeline=Pipeline(
                steps=[
                    #handle missing values and scale the data
                    ('imputer',SimpleImputer(strategy='median')),
                    ('std_scaler',StandardScaler())])
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
                    ('std_scaler',StandardScaler())])
            
            logging.info("Catagorical and Numerical Pipeline created")
            logging.info("Numerical Columns: {}".format(numerical_col))
            logging.info("Catagorical Columns: {}".format(categorical_col))
            
            preprocessor=ColumnTransformer(
                transformers=
                [
                    ('num',num_pipeline,numerical_col),
                    ('cat',cat_pipeline,categorical_col)
                ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test dataset as dataframe")
            
            preprocessor=self.get_data_transformation_object()
            
            logging.info("Obtain preprocessor object")
            
            target_column_name='G3'
            
            input_features_train_df=train_df.drop(columns=[target_column_name], axis=1, inplace=False)
            target_feature_train_df=train_df[target_column_name]
            
            input_features_test_df=test_df.drop(columns=[target_column_name], axis=1, inplace=False)
            target_feature_test_df=test_df[target_column_name]
            

            # input_feature_train_arr=preprocessor.fit_transform(input_features_train_df)
            # input_feature_test_arr=preprocessor.transform(input_features_test_df)
            
            input_feature_train_arr=preprocessor.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessor.transform(input_features_test_df)
            
            logging.info("Apply preprocessing object on training df and testing df")
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor)
            
            logging.info("saved preprocessor object")
            
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)
        