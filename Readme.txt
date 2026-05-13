Email Spam Classifier — Production-Ready ML Pipeline

Overview مشروع تصنيف الرسائل المزعجة (Spam) هو نظام متكامل (End-to-End
ML Pipeline) تم تصميمه للتحول من مجرد تجربة إلى منتج حقيقي باستخدام
أدوات MLOps و Software Engineering الحديثة.

The Transformation (Before vs. After) يمثل المشروع نقلة من العمل
العشوائي إلى العمل المنظم:

-   Workflow: النهج التقليدي: ملفات Notebook متداخلة المشروع: كود
    Modular منظم (Scripts)

-   Reproducibility: النهج التقليدي: صعوبة إعادة إنتاج النتائج المشروع:
    DVC Pipeline لضمان تكرار النتائج

-   Tracking: النهج التقليدي: ضياع التجارب المشروع: تتبع عبر YAML و
    DVCLive

-   Serving: النهج التقليدي: لا يمكن استخدام الموديل بسهولة المشروع: API
    باستخدام FastAPI

-   Portability: النهج التقليدي: مشاكل البيئة المشروع: Docker لتشغيل
    المشروع في أي مكان

Model Benchmarking & Selection تم اختبار 10 خوارزميات مختلفة:

-   Extra Trees: Accuracy 97.74%, F1 90.68%
-   Random Forest: Accuracy 97.64%, F1 90.16%
-   XGBoost: Accuracy 97.05%, F1 87.28%
-   Naive Bayes: Accuracy 97.24%, F1 88.52%

Final Decision: تم اختيار Random Forest مع TF-IDF (500 features) لتحقيق
أفضل توازن بين Precision و Recall.

Final Metrics: - Accuracy: 97.71% - Precision: 94.20% - Recall: 88.43% -
AUC: 98.46% - F1-Score: 91.22%

Technical Architecture - DVC: إدارة مراحل المشروع - FastAPI: إنشاء API -
Docker: تشغيل المشروع في أي بيئة

Tech Stack - ML/NLP: Scikit-learn, SpaCy, TF-IDF - MLOps: DVC, Git,
YAML - Backend: FastAPI, Uvicorn - Infrastructure: Docker

Run Locally 1. Install dependencies: pip install -r requirements.txt

2.  Run pipeline: dvc repro

3.  Start API: uvicorn deployment.app:app –reload

4.  Run Docker: docker build -t spam-app . docker run -p 8000:8000
    spam-app

Author Abdelwahab Amr Software Engineering Student | AI & MLOps
