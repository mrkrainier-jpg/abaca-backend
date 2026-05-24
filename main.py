import os
import gdown
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io


DISEASE_MODEL_ID = "1aXVmndI_JHNb7PjMtGAbbEv9MToW700L"
FIBER_MODEL_ID = "1Mep2V55UrM70gJmmmYEdJToxr4hT9xm_"

def download_model_if_missing(file_name, file_id):
    if not os.path.exists(file_name):
        print(f"Downloading {file_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
    else:
        print(f"{file_name} already exists.")


download_model_if_missing("model.h5", DISEASE_MODEL_ID)
download_model_if_missing("fiber_model.h5", FIBER_MODEL_ID)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

disease_model = tf.keras.models.load_model("model.h5")
disease_classes = ["Bunchy Top", "Mosaic", "Normal"]

fiber_model = tf.keras.models.load_model("fiber_model.h5")
fiber_classes = ["EF", "S2", "S3"]

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