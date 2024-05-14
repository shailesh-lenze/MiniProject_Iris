import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models
from sklearn import metrics

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest Classifier": RandomForestClassifier()
            }
            
            # New parameters for finetuning the RandomForestClassifier
            #params={
            #    "Random Forest Classifier":{
            #        'model__n_estimators': [50,100,200,300,400,500],
            #        'model__max_depth': [None,5,10,15,20]
            #   }
            #}
            
            
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            # #model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # To get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            # To get best model name from dic
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            accuracy=metrics.accuracy_score(y_test, predicted)
            return accuracy
            
        except Exception as e:
            raise CustomException(e,sys)
