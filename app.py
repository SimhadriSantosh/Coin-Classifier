from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app_predict = FastAPI()

origins = [
    "*",
]

# Add CORS middleware to the app
app_predict.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("Models/Model_VGG16")

with open('cat_to_name.json', 'r') as json_file:
    cat_2_name = json.load(json_file)

label_no = pd.read_csv('Label_no.csv')

@app_predict.get('/JSM')
async def hello():
    return "Jai Santoshi Maata"

def read_file(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image / 255.0 - mean) / std
    return image

@app_predict.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file(await file.read())
    image_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(image_batch)
    idx = np.argmax(prediction)
    label_idx = str(label_no.loc[idx,'Label_no'])
    predicted_class = cat_2_name[label_idx]
    return predicted_class

if __name__ == '__main__':
    uvicorn.run(app_predict,host = 'localhost',port =8000)