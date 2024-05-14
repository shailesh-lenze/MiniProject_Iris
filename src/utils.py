import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn import metrics
#from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
#def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            #para=param[list(models.keys())[i]]
            
            #gs=GridSearchCV(model, para,cv=3) # scoring='f1_macro', n_jobs=-1, verbose=1
            #gs.fit(X_train, y_train)
            
            #model.set_params(**gs.best_params)
            model.fit(X_train, y_train) # Train model
            
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = metrics.accuracy_score(y_train_pred, y_train)
            
            test_model_score = metrics.accuracy_score(y_test_pred, y_test)
            
            report[list(models.keys())[i]]=(test_model_score)
            #report[list(models.keys())[i]]=(test_model_score,gs.best_params_)
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)