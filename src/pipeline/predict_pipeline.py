import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
from src.exception import CustomException

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
class IrisModel:
    def __init__(self):
        #self.df = pd.read_csv('iris.csv')
        self.model = joblib.load('artifacts/model.pkl')
        #self.preprocessor=joblib.load('artifacts/preprocessor.pkl')
        
        #self.model_fname_ = 'iris_model.pkl'
        #try:
        #    self.model = joblib.load(self.model_fname_)
        #except:
        #    pass
        
        #except Exception as _:
        #    self.model = self._train_model()
        #    joblib.dump(self.model, self.model_fname_)
            

    #def _train_model(self):
    #        X=self.df.drop('species', axis=1)
    #        y=self.df['species']
    #        rfc=RandomForestClassifier()
    #        model = rfc.fit(X.values, y)
    #        return model
        
    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        try:
            data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
            #data_in=pd.DataFrame({[sepal_length], [sepal_width], [petal_length], [petal_width]})
            #data_scaled=self.preprocessor.transform(data_in)
            #prediction = self.model.predict(data_scaled)
            #probability= self.model.predict_proba(data_scaled).max()
            prediction = self.model.predict(data_in)
            probability= self.model.predict_proba(data_in).max()
            return prediction[0], probability
        
        except Exception as e:
            raise CustomException(e,sys)
    