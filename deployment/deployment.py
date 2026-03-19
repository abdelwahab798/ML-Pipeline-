import logging
import os 
import joblib
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import transform_text
import pandas as pd

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("pipeline-production")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"pipeline-production.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)

def load_artifacts(model_path,vectorizer_path)->tuple:
    try:
        model=joblib.load(model_path)
        vectorizer=joblib.load(vectorizer_path)
        logger.debug("The artifacts is loaded")
        return model,vectorizer
    except Exception as e:
        logger.error('Error during pipeline : %s', e)
        raise

def predict(text:str,model,vectorizer):
    try:
        cleaned_text = transform_text(text)
        logger.debug("Text is transformed")
        
        text_vec = pd.DataFrame(
        vectorizer.transform([cleaned_text]).toarray(),
        columns=vectorizer.get_feature_names_out())

        prediction = model.predict(text_vec)[0]
        label = "spam" if prediction == 1 else "not spam"
        logger.debug("Prediction is done: %s", label)
        
        return label
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise
    
def main():
    model,vectorizer=load_artifacts(r"deployment\model.pkl",r"deployment/vectorizer.pkl")
    text = "Congratulations! You won a free prize. Click here to claim now!"
    label = predict(text=text, model=model, vectorizer=vectorizer)
    print(f"Prediction: {label}")
    logger.debug("Pipeline is done")

if __name__ == "__main__":
    main()



