import os 
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
model_path=r"model.pkl"
with open(model_path,"rb")as f:
            model=pickle.load(f)
app = FastAPI(title="Email Spam Classifier API")

# Input schema
class EmailRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Spam Classifier API is running 🚀"}


@app.post("/predict")
def predict(req: EmailRequest):
    pred = model.predict([req.text])[0]

    if pred == 1:
        return {"result": "Spam 🚨"}
    else:
        return {"result": "Not Spam ✅"}

