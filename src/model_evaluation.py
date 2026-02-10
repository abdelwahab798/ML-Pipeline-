import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path:str,):
    try:
        with open(model_path,"rb")as f:
            model=pickle.load(f)
        logger.debug("Model is loadded")
        return model
    except Exception as e :
        logger.error("we have error in model load: %s",e)
        raise

def load_data():
    try:
        data_test=pd.read_csv(r"C:\Users\nice\Desktop\Mlops\ML-Pipeline-\data\processed\test_tfidf.csv")
        x_test=data_test.drop(columns=["target"])
        y_test=data_test["target"]
        logger.debug("load data is done")
        return x_test,y_test
    except Exception as e:
        logger.error("we have in load data: %s",e)
        raise

def evaluate_model(x_test:pd.DataFrame,y_test:pd.DataFrame,model)->dict:
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        model=load_model(r"C:\Users\nice\Desktop\Mlops\ML-Pipeline-\models\model.pkl")
        x_test,y_test=load_data()
        metircs=evaluate_model(x_test,y_test,model)
        save_metrics(metircs,'reports/metrics.json')
        logger.debug("all done")
    except Exception as e:
        logger.error("we have error in main: %s",e)

if __name__ == '__main__':
    main()

    

    
    