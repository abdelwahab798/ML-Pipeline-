import logging
import os 
import spacy as sp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
nlp = sp.load("en_core_web_sm")


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)

def transform_text(text):
   
    text=text.lower()
    doc=nlp(text)
    tokens=[token.lemma_
        for token in doc 
        if not token.is_stop
        and not token.is_punct
    ]
    return " ".join(tokens)
    

def preprocesing(df:pd.DataFrame,text_column="text",target_column="target")->pd.DataFrame:
    try:
        endcoder=LabelEncoder()
        df[target_column]=endcoder.fit_transform(df[target_column])
        logger.debug("encoding is done")
        df.drop_duplicates(inplace=True)
        logger.debug("drop duplicates is done")
        df[text_column]=df[text_column].apply(transform_text)
        logger.debug("transforaming is done")
        return df
    except KeyError as e:
        logger.error("we have error %s",e)
    except Exception as e:
        logger.error("we have error %s",e)
        raise
def save_data(train_data:pd.DataFrame,test_data)->None:
    try:
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        logger.debug("The new floder for a preocessed data is done")
        train_data.to_csv(os.path.join(data_path,"processed_train.csv"),index=True)
        test_data.to_csv(os.path.join(data_path,"processed_test.csv"),index=True)
        logger.debug("data is saved  ")
    except Exception as e:
      logger.error( "we have error %s",e)
      raise

def main(text_column="text",target_column="target"):
    train_data=pd.read_csv(r"C:\Users\nice\Desktop\Mlops\ML-Pipeline-\data\raw\train_data.csv")
    test_data=pd.read_csv(r"C:\Users\nice\Desktop\Mlops\ML-Pipeline-\data\raw\test_data.csv")
    train_data_pro=preprocesing(train_data)
    test_data_pro=preprocesing(test_data)
    save_data(train_data_pro,test_data_pro)

if __name__=="__main__":
    main()





    
    
    