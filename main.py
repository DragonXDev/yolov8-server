from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = YOLO("deepfashion2_yolov8s-seg.pt")  # Or your YOLOv11 model

def classify_color_from_rgb(r, g, b):
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if v < 40:
        return 'black'
    elif s < 30 and v > 200:
        return 'white'
    elif s < 40:
        return 'gray'
    elif h < 15 or h >= 165:
        return 'red'
    elif 15 <= h < 35:
        return 'orange'
    elif 35 <= h < 65:
        return 'yellow'
    elif 65 <= h < 90:
        return 'lime'
    elif 90 <= h < 150:
        return 'green'
    elif 150 <= h < 190:
        return 'cyan'
    elif 190 <= h < 250:
        return 'blue'
    elif 250 <= h < 290:
        return 'purple'
    elif 290 <= h < 330:
        return 'magenta'
    else:
        return 'unknown'

def get_dominant_color(crop):
    resized = cv2.resize(crop, (50, 50))
    avg_color = np.mean(resized.reshape(-1, 3), axis=0).astype(int)
    b, g, r = avg_color
    return classify_color_from_rgb(r, g, b)
    
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    annotated_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_img)
    font = ImageFont.load_default()

    results = model(img)[0]
    output = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, f"unknown_{cls_id}")

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        color = get_dominant_color(crop)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label}, {color}", fill="white", font=font)

        output.append({
            "label": label,
            "color": color,
            "box": [x1, y1, x2, y2]
        })

    return {"results": output}
