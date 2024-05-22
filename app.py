# File: /app.py
# This file will contain the code for the FastAPI application
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class HealthData(BaseModel):
    data: list

# Load models
heart_model = joblib.load('models/heart_model.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')

@app.post('/predict_heart')
def predict_heart(data: HealthData):
    prediction = heart_model.predict([data.data])
    return {"prediction": int(prediction[0])}

@app.post('/predict_diabetes')
def predict_diabetes(data: HealthData):
    prediction = diabetes_model.predict([data.data])
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
