# 📧 Email Spam Classifier — ML Pipeline


A production-ready end-to-end ML pipeline that classifies emails as **spam** or **not spam**. Built with a full MLOps workflow including data versioning, experiment tracking, and live deployment.

- Built an end-to-end ML pipeline (data ingestion → preprocessing → training → evaluation → deployment)
- Used **Git**, **DVC**, and **YAML** for version control, experiment tracking, and configuration management
- Implemented **logging**, **exception handling**, and **cloud integration** for scalable, production-ready deployment

---

## 🚀 Live Demo

👉 **[Try it here](https://huggingface.co/spaces/Kahfaji/EmailClassifier)**

Paste any email content and get an instant classification.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **97.70%** |
| Precision | **94.20%** |
| Recall | **88.43%** |
| AUC | **98.46%** |
| F1 | **91.22%** |

> Model: Random Forest Classifier trained on TF-IDF features with spaCy preprocessing.

---

## 🏗️ Project Structure

```
├── data/
│   ├── raw/              # Train/test split after ingestion
│   ├── interim/          # Preprocessed text data
│   └── processed/        # TF-IDF feature matrices
├── deployment/
│   ├── app.py            # FastAPI application
│   ├── deployment.py     # Production inference pipeline
│   ├── model.pkl         # Trained Random Forest model
│   └── vectorizer.pkl    # Fitted TF-IDF vectorizer
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
├── models/               # Saved model artifacts
├── reports/              # Evaluation metrics (JSON)
├── logs/                 # Pipeline execution logs
├── params.yaml           # Hyperparameters & config
├── dvc.yaml              # DVC pipeline definition
└── Dockerfile
```

---

## ⚙️ ML Pipeline

The pipeline is fully orchestrated with **DVC** and consists of 5 stages:

```
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation
```

### Stage 1 — Data Ingestion
- Loads raw SMS spam dataset (5,572 records)
- Drops unused columns and renames to `text` / `target`
- Splits into train/test sets (configurable via `params.yaml`)

### Stage 2 — Data Preprocessing
- Label encodes target: `ham=0`, `spam=1`
- Removes duplicates
- Applies NLP preprocessing via **spaCy**:
  - Lowercasing
  - Lemmatization
  - Stop word & punctuation removal

### Stage 3 — Feature Engineering
- Applies **TF-IDF Vectorization** (configurable `max_features`)
- Fits vectorizer on training data only
- Saves fitted `vectorizer.pkl` for production use

### Stage 4 — Model Building
- Trains a **Random Forest Classifier**
- Hyperparameters configured via `params.yaml`
- Saves model to `models/` and `deployment/`

### Stage 5 — Model Evaluation
- Evaluates on held-out test set
- Tracks metrics with **DVCLive**
- Saves results to `reports/metrics.json`

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | scikit-learn, Random Forest |
| NLP | spaCy, TF-IDF |
| Pipeline | DVC |
| Experiment Tracking | DVCLive |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |

---

## 🔧 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/abdelwahab798/ML-Pipeline-.git
cd ML-Pipeline-
```

### 2. Install dependencies
```bash
pip install -r deployment/requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the full pipeline
```bash
dvc repro
```

### 4. Start the API
```bash
uvicorn deployment.app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🐳 Run with Docker

```bash
docker build -t email-classifier .
docker run -p 7860:7860 email-classifier
```

---

## 📁 Configuration

All pipeline parameters are controlled from `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 500

model_building:
  n_estimators: 50
  random_state: 42
```

---

## 👤 Author

**Abdelwahab Amr**
- 🎓 Software Engineering Student — Delta Technology University
- 💼 Data Scientist | AI & ML
- 🤗 [Hugging Face](https://huggingface.co/Kahfaji)
- 🐙 [GitHub](https://github.com/abdelwahab798)
