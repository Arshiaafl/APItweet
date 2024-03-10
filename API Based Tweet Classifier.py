#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

# Load the trained model
model = load_model('trained_model.h5')

# Load the Tokenizer and LabelEncoder
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = joblib.load(tokenizer_file)

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = joblib.load(label_encoder_file)

app = FastAPI()

class PredictionInput(BaseModel):
    text: List[str]

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Tokenize and pad the input text
        text_sequences = tokenizer.texts_to_sequences(data.text)
        padded_sequences = pad_sequences(text_sequences, padding='post', maxlen=model.input_shape[1])

        # Make predictions using the trained model
        predictions = model.predict(padded_sequences)
        predicted_labels = np.argmax(predictions, axis=1)

        # Convert numerical labels back to original text labels
        predicted_emotions = label_encoder.inverse_transform(predicted_labels)

        return {"predictions": predicted_emotions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# In[ ]:




