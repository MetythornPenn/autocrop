import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import onnxruntime as ort
from autocrop_kh import autocrop, load_autocrop_model

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, device
    model_path = None
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    model = load_autocrop_model(model_path, device)


def process_image(image_data: bytes):
    print(f"Length of image_data: {len(image_data)}")
    nparr = np.frombuffer(image_data, np.uint8)
    print(f"First few bytes of image_data: {image_data[:10]}")
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. The input data may not be a valid image.")
    cropped_img = autocrop(np_image=img, trained_model=model, device=device)
    _, img_encoded = cv2.imencode('.jpg', cropped_img)
    return img_encoded.tobytes()



@app.post("/crop-image/")
async def crop_image(file: UploadFile = File(...)):
    image_data = await file.read()
    cropped_image = process_image(image_data)
    return StreamingResponse(io.BytesIO(cropped_image), media_type="image/jpeg")
