import pandas as pd 
import logging 
import os 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import yaml
import joblib


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("feature_engineer")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"feature_engineer.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
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

def transform_x(df_train:pd.DataFrame,df_test:pd.DataFrame,max_features=50)->tuple:
    try:
        tf=TfidfVectorizer(max_features=max_features)
        x_train = df_train["text"].fillna("").astype(str).values
        x_test = df_test["text"].fillna("").astype(str).values
        y_train=df_train["target"].fillna("").values
        y_test=df_test["target"].values
        logger.debug("spilt data is done")

        x_train_vec=tf.fit_transform(x_train)
        x_test_vec=tf.transform(x_test)
        logger.debug("transform data is done")

        feature_names = tf.get_feature_names_out()
        final_train=pd.DataFrame(x_train_vec.toarray(),columns=feature_names)
        final_train["target"]=y_train
        final_test=pd.DataFrame(x_test_vec.toarray(),columns=feature_names)
        final_test["target"]=y_test
        logger.debug("New Dataframe is exit")

        joblib.dump(tf,r"deployment\vectorizer.pkl")
        logger.debug("Vectorizer is saved ")

        return final_train,final_test
    except Exception as e:
        logger.exception("we have error: %s",e)
        raise

    
    

def save_data(df:pd.DataFrame,file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("Data is saved")
    except Exception as e:
        logger.exception("we have error: %s",e)
        raise


def main():
    params=load_parmas("params.yaml")
    max_features=params["feature_engineering"]["max_features"]
    df=pd.read_csv(r"data\interim\processed_train.csv")
    df2=pd.read_csv(r"data\interim\processed_test.csv")
    train_df,test_df=transform_x(df,df2,max_features=max_features)
    save_data(train_df,os.path.join("./data", "processed", "train_tfidf.csv"))
    save_data(test_df,os.path.join("./data", "processed", "test_tfidf.csv"))
    logger.debug("All fuctions is done and data is ready and save")

if __name__=="__main__":
    main()
    




