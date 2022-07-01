from fastapi import FastAPI, Depends, Request
from schemas import Churn,ChurnDriftInput
import os
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
import pandas as pd
from scipy.stats import ks_2samp
import models
from database import engine, get_db
from sqlalchemy.orm import Session
from flask import Response
from fastapi.responses import JSONResponse


# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

# Learn, decide and get model from mlflow model registry
model_name = "RFChurnModel"
model_version = 1
model = load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

app = FastAPI()

# Creates all the tables defined in models module
models.Base.metadata.create_all(bind=engine)

def insert_churn(request, prediction, client_ip, db):
    new_churn= models.Churn(
        CustomerId=request["CustomerId"],
        Surname=request['Surname'],
        CreditScore=request['CreditScore'],
        Geography=request['Geography'],
        Gender=request['Gender'],
        Age=request['Age'],
        Tenure=request['Tenure'] ,
        Balance=request['Balance'],
        NumOfProducts=request['NumOfProducts'],
        HasCrCard=request['HasCrCard'],
        IsActiveMember=request['IsActiveMember'] ,
        EstimatedSalary=request['EstimatedSalary'],
        prediction=prediction,
        client_ip=client_ip
    )

    db.add(new_churn)
    db.commit()
    db.refresh(new_churn)
    return new_churn

# Note that model is coming from mlflow
def make_churn_prediction(model, request):
    ##parse input from request
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
    
    #Gender,
    # Make an input vector
    churn=[[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]]

    # Predict
    prediction = model.predict(churn)
 
    return {int(prediction[0])}


# IsChurn Prediction endpoint
@app.post("/prediction/churn")
async def predict_churn(request: Churn,fastapi_req: Request,db: Session = Depends(get_db)):
    prediction = make_churn_prediction(model, request.dict())
    db_insert_record = insert_churn(request=request.dict(), prediction=prediction,client_ip=fastapi_req.client.host,db=db)
                                          
    
    print(prediction)

    return {'prediction': prediction,"db_record": db_insert_record}
    
    
# Object agnostic drift detection function
def detect_drift(data1, data2):
    ks_result = ks_2samp(data1, data2)
    if ks_result.pvalue < 0.05:
       return "Drift exits"
    else:
       return "No drift"
       
       
       
# IsChurn drift detection endpoint
@app.post("/drift/churn")
async def detect(request: ChurnDriftInput):
    # Select training data
    train_df =  pd.read_sql("select * from churn_train", engine)

    # Select predicted data last n days
    prediction_df = pd.read_sql(f"""select * from churn 
                                    where prediction_time >
                                    current_date - {request.n_days_before}""",
                                engine)
                                

    CreditScore_drift=detect_drift(train_df.CreditScore, prediction_df.CreditScore)
    Age_drift=detect_drift(train_df.Age, prediction_df.Age)
    Tenure_drift=detect_drift(train_df.Tenure, prediction_df.Tenure)
    Balance_drift=detect_drift(train_df.Balance, prediction_df.Balance)
    NumOfProducts_drift=detect_drift(train_df.NumOfProducts, prediction_df.NumOfProducts)
    HasCrCard_drift=detect_drift(train_df.HasCrCard, prediction_df.HasCrCard)
    IsActiveMember_drift=detect_drift(train_df.IsActiveMember, prediction_df.IsActiveMember) 
    EstimatedSalary_drift=detect_drift(train_df.EstimatedSalary, prediction_df.EstimatedSalary)


    return { "CreditScore_drift": CreditScore_drift, "Age_drift":   Age_drift,'Tenure_drift':Tenure_drift,'Balance_drift':Balance_drift,'NumOfProducts_drift':NumOfProducts_drift,'HasCrCard_drift':HasCrCard_drift,'IsActiveMember_drift':IsActiveMember_drift,'EstimatedSalary_drift':EstimatedSalary_drift}







    
