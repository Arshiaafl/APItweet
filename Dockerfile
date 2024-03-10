# Use an official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY ["API Based Tweet Classifier.py", "./"]

COPY trained_model.h5 /app/

COPY tokenizer.pkl /app/

COPY label_encoder.pkl /app/

# Install additional dependencies
RUN pip install fastapi uvicorn pydantic tensorflow numpy joblib

# Specify the command to run your Python script
CMD ["python", "API Based Tweet Classifier.py"]
