import os
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import yaml
import joblib


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_parmas(file_params:str)->dict:
    try:
        with open(file_params,"r") as file:
            parmas=yaml.safe_load(file)
        logger.debug("load parmas is done")
        return parmas
    except Exception as e:
        logger.error("we have error in load parmas: %s",e)
        raise


def load_data(df_train:pd.DataFrame,df_test:pd.DataFrame)->tuple:
    try:
        x_train=df_train.drop(columns=["target"])
        y_train=df_train["target"]
        x_test=df_test.drop(columns=["target"])
        logger.debug("spilt data is done")

        return x_train,y_train,x_test
    except Exception as e:
        logger.error("we have error in splilting data: %s",e)
        raise

def model_train(model:BaseEstimator,x_train:pd.DataFrame,y_train:pd.DataFrame):
    try:
        model.fit(x_train,y_train)
        logger.debug("fit model is done")
        return model
    except Exception as e:
        logger.error("we have error in fiting and predict: %s",e)
        raise

def save_model(model,file_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise
   

    
def main():
    try:
        params= load_parmas("params.yaml")
        radom=params["model_building"]["random_state"]
        n=params["model_building"]["n_estimators"]
        model= RandomForestClassifier(random_state=radom,n_estimators=n)
        df_train=pd.read_csv(r"data\processed\train_tfidf.csv")
        df_test=pd.read_csv(r"data\processed\test_tfidf.csv")
        x_train,y_train,x_test=load_data(df_train,df_test)
        model=model_train(model=model,x_train=x_train,y_train=y_train)
        model_path=r"models/model.pkl"
        model_path2=r"deployment/model.pkl"
        save_model(model, model_path)
        save_model(model,model_path2)
        logger.debug("done")
    except Exception as e:
        logger.error("we have error in main: %s",e)
if __name__=="__main__":
    main()




        


