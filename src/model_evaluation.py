import os
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,f1_score
import logging
from dvclive import Live
import yaml
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

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise

def load_model(model_path:str,):
    try:
        model=joblib.load(model_path)
        logger.debug("Model is loadded")
        return model
    except Exception as e :
        logger.error("we have error in model load: %s",e)
        raise

def load_data():
    try:
        data_test=pd.read_csv(r"data\processed\test_tfidf.csv")
        x_test=data_test.drop(columns=["target"])
        y_test=data_test["target"]
        logger.debug("load data is done")
        return x_test,y_test
    except Exception as e:
        logger.error("we have in load data: %s",e)
        raise

def evaluate_model(x_test:pd.DataFrame,y_test:pd.DataFrame,model)->tuple:
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1= f1_score(y_test, y_pred)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1': f1
        }
        logger.debug('Model evaluation metrics calculated')
        return y_pred,metrics_dict
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
        model=load_model(r"models\model.pkl")
        params=load_params(r"params.yaml")
        x_test,y_test=load_data()
        y_pred,metircs=evaluate_model(x_test,y_test,model)
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_pred))
            live.log_metric('precision', precision_score(y_test, y_pred))
            live.log_metric('recall', recall_score(y_test, y_pred))
            live.log_params(params)
        
        save_metrics(metircs,'reports/metrics.json')
        logger.debug("all done")
    except Exception as e:
        logger.error("we have error in main: %s",e)

if __name__ == '__main__':
    main()

    

    
    