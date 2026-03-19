FROM python:3.10-slim

WORKDIR /app

# Copy all needed files
COPY deployment/ ./deployment/
COPY src/ ./src/

RUN pip install --no-cache-dir -r deployment/requirements.txt
RUN python -m spacy download en_core_web_sm

EXPOSE 7860

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "7860"]