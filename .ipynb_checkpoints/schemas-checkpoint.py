from pydantic import BaseModel


class Churn(BaseModel):
    CustomerId:int
    Surname:str
    CreditScore:int  
    Geography:str
    Gender:str
    Age:int
    Tenure:int 
    Balance:float
    NumOfProducts:int 
    HasCrCard:int 
    IsActiveMember:int 
    EstimatedSalary:float
    Exited:int

    class Config:
        schema_extra = {
            "example": {
                'CustomerId':15634602,
                'Surname':'Hargrave',
                'CreditScore':619 , 
                'Geography':'France',
                'Gender':'Female',
                'Age':42,
                'Tenure':2,
                'Balance':0,
                'NumOfProducts':1 ,
                'HasCrCard':1,
                'IsActiveMember':1, 
                'EstimatedSalary':101348,
                'Exited':1
                
            }
        }



