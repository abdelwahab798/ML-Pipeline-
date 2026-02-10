import pandas as pd 
import logging 
import os 
from sklearn.model_selection import train_test_split

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data is excuting")
        return df 
    except pd.errors.ParserError as e :
        logger.error("Data filed in excuting: %s",e)
    except Exception as e:
        logger.error("anexpected error: %s",e)
        raise

def simple_pre(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
        df.rename(columns={"v1":"target","v2":"text"},inplace=True)
        logger.debug("drop preprocesing done ")
        return df 
    except KeyError as e:
        logger.debug("No such columns: %s",e)
        raise
    except Exception as e :
        logger.debug("anexpected error: %s",e)
        raise

def save_data(trian_data:pd.DataFrame,test_data:pd.DataFrame,data_url:str)->None:
    try:
        raw_data=os.path.join(data_url,"raw")
        os.makedirs(raw_data,exist_ok=True)
        trian_data.to_csv(os.path.join(raw_data,"train_data.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data,"test_data.csv"),index=False)
        logger.debug("we save all data and split it  to: %s ",raw_data)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    
def main():
    try:
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        test_size=0.2
        df=load_data(data_url=data_path)
        df_final=simple_pre(df)
        train_data,test_data=train_test_split(df_final,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_url='./data')
        logger.debug("data ingetion is done ")
    except Exception:
        logger.exception("Error occurred in data ingestion pipeline")

if __name__=="__main__":
    main()
