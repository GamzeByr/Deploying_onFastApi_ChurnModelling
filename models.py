from database import Base
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func


class Churn(Base):
    __tablename__ = "churn"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, autoincrement=True, primary_key=True)
    CustomerId = Column(Integer)
    Surname=Column(String(100))
    CreditScore=Column(Integer) 
    Geography=Column(String(20))
    Gender=Column(String(20))
    Age=Column(Integer)
    Tenure=Column(Integer) 
    Balance=Column(Float)
    NumOfProducts=Column(Integer)
    HasCrCard=Column(Integer) 
    IsActiveMember=Column(Integer)
    EstimatedSalary=Column(Float)
    prediction = Column(Float)
    prediction_time = Column(DateTime(timezone=True), server_default=func.now())
    client_ip = Column(String(20))


class ChurnTrain(Base):
    __tablename__ = "churn_train"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, autoincrement=True, primary_key=True)
    CustomerId = Column(Integer)
    Surname=Column(String(100))
    CreditScore=Column(Integer) 
    Geography=Column(String(20))
    Gender=Column(String(20))
    Age=Column(Integer)
    Tenure=Column(Integer) 
    Balance=Column(Float)
    NumOfProducts=Column(Integer)
    HasCrCard=Column(Integer) 
    IsActiveMember=Column(Integer)
    EstimatedSalary=Column(Float)
    Exited=Column(Float,nullable=True)
