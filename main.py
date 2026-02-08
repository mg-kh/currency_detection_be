from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

origins = [
    'https://thai-currency-detection.vercel.app'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = tf.keras.models.load_model("model/currency_classification_model_v2.keras")
IMG_SIZE = (224,224)

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0   # normalize like training
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, C)
    return img_array

class_indices = {'100': 0, '1000': 1, '20': 2, '50': 3, '500': 4}
index_to_label = {v: k for k, v in class_indices.items()}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/predict-image')
async def predict_image(file:UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        return {"error" : 'File must be an image'}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    processed_image = preprocess_image(image) 
    predictions = model.predict(processed_image)
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    predicted_label = index_to_label[predicted_index]

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
    }
