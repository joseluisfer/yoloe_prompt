from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Cargar modelo al inicio
model = YOLOE("yoloe-26x-seg.pt")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text_prompt: str = Form(...)
):
    # Leer imagen
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Configurar clases
    classes = [c.strip() for c in text_prompt.split(",")]
    model.set_classes(classes)
    
    # Predecir
    results = model.predict(np.array(img), verbose=False)
    
    # Extraer detecciones
    detections = []
    if results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names
        
        for i in range(len(boxes)):
            detections.append({
                "class": names[cls_ids[i]],
                "confidence": float(confs[i]),
                "bbox": boxes[i].tolist()
            })
    
    return {"detections": detections}

@app.get("/health")
def health():
    return {"status": "ok"}
