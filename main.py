from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("model.pt")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_array = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model(img)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    labels = [model.names[int(cls)] for cls in results.boxes.cls]

    return {"labels": labels, "boxes": boxes}
