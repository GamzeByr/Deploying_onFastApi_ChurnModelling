from fastapi import FastAPI, Depends, Request
from schemas import Churn
import os
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

# Learn, decide and get model from mlflow model registry
model_name = "ChurnRFModel"
model_version = 2
model = load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

app = FastAPI()

# Note that model is coming from mlflow
def make_advertising_churn(model, request):
    # parse input from request
    CustomerId=request["CustomerId"]
    Surname=request['Surname']
    CreditScore=request['CreditScore']
    Geography=request['Geography']
    Gender=request['Gender']
    Age=request['Age']
    Tenure=request['Tenure'] 
    Balance=request['Balance']
    NumOfProducts=request['NumOfProducts']
    HasCrCard=request['HasCrCard']
    IsActiveMember=request['IsActiveMember'] 
    EstimatedSalary=request['EstimatedSalary']
    Exited=request['Exited']

    # Make an input vector
    churn=[[CustomerId,Surname,CreditScore,Geography,Gender,Age,enure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited]]

    # Predict
    prediction = model.predict(churn)

    return prediction[0]

# Advertising Prediction endpoint
@app.post("/prediction/churn")
async def predict_churn(request: Churn):
    prediction = make_churn_prediction(model, request.dict())

    return {"prediction": prediction}
