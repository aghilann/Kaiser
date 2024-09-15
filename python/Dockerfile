# Dockerfile

FROM python:3.9-slim

RUN pip install scikit-learn pandas joblib minio

COPY california_housing.py /app/california_housing.py
COPY task_executor.py /app/task_executor.py

WORKDIR /app

CMD ["python", "task_executor.py"]