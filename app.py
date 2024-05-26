import uvicorn
from fastapi import FastAPI
from src.pipeline.predict_pipeline import IrisModel, IrisSpecies

app = FastAPI()
model = IrisModel()

@app.get('/')
def home():
    return{"Hello": "Welcome to the ML app"}

@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)