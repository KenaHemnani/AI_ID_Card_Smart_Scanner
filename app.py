from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inference import run_inference
from ultralytics import YOLO
import shutil
import uuid
import os
import cv2
import base64

app = FastAPI()
MODEL_PATH = "./checkpoint/best_doc_seg_yolo.pt"
model = YOLO(MODEL_PATH)

def encode_image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

@app.post("/process/")
async def process_document(
    image: UploadFile = File(...)
):
    # Save uploaded image to a temporary location
    tmp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(tmp_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Run inference
    processed_img, rotation_angle = run_inference(model, tmp_filename)

    # Remove temp input image
    os.remove(tmp_filename)

    # Encode image as base64
    encoded_img = encode_image_to_base64(processed_img)

    # Return JSON response with base64 image and rotation
    return JSONResponse(content={
        "rotation_angle": rotation_angle,
        "image_base64": encoded_img
    })
