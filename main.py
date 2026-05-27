import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

disease_model = tf.keras.models.load_model("disease_model.keras", compile=False)
disease_classes = ["Bunchy Top", "Mosaic", "Normal"]

fiber_model = tf.keras.models.load_model("fiber_model.keras", compile=False)
fiber_classes = ['EF', 'G', 'H', 'I', 'JK', 'M1', 'S2', 'S3', 'Y1', 'Y2']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def read_root():
    return {"status": "Abaca API is Live"}

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        predictions = disease_model.predict(image)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        return {"prediction": disease_classes[class_idx], "confidence": confidence}
    except Exception as e:
        return {"prediction": f"Error: {str(e)}", "confidence": 0.0}

@app.post("/predict_fiber")
async def predict_fiber(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        predictions = fiber_model.predict(image)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        return {"prediction": fiber_classes[class_idx], "confidence": confidence}
    except Exception as e:
        return {"prediction": f"Error: {str(e)}", "confidence": 0.0}