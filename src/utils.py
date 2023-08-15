import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

class CustomStratifiedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X, y=None, groups=None):
        random_state = check_random_state(self.random_state)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        class_indices = {}
        for cls in unique_classes:
            class_indices[cls] = np.where(y == cls)[0]
        
        for _ in range(self.n_splits):
            train_indices = []
            test_indices = []
            
            for cls, count in zip(unique_classes, class_counts):
                indices = class_indices[cls]
                random_state.shuffle(indices)
                split_point = int(count / self.n_splits)
                
                test_indices.extend(indices[:split_point])
                train_indices.extend(indices[split_point:])
            
            if self.shuffle:
                random_state.shuffle(train_indices)
                random_state.shuffle(test_indices)
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            # Use the custom cross-validator in GridSearchCV
            gs = GridSearchCV(model, param, cv=CustomStratifiedKFold(n_splits=3))
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            #train model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            logging.info(f"Model: {list(models.keys())[i]}")
        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = dill.load(f)
        return obj
    except Exception as e:
        raise CustomException(e,sys)