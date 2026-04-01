from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLOE # Ahora sí, usando la clase oficial de la v8.4
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Cargar el modelo YOLOE-X de segmentación
model = YOLOE("yoloe-26x-seg.pt")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text_prompt: str = Form(...)
):
    # 1. Procesar imagen de la escena
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    
    # 2. Configurar el vocabulario dinámico (Text Prompt)
    # YOLOE permite definir qué buscar en tiempo real
    classes = [c.strip() for c in text_prompt.split(",")]
    model.set_classes(classes)
    
    # 3. Inferencia
    results = model.predict(img_array, verbose=False)
    
    detections = []
    if results and len(results) > 0:
        res = results[0]
        # Procesar detecciones de segmentación
        if res.boxes:
            for i in range(len(res.boxes)):
                box = res.boxes[i]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                det = {
                    "class": classes[cls_id] if cls_id < len(classes) else "unknown",
                    "confidence": round(conf, 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
                }
                
                # Si quieres incluir la máscara de segmentación:
                if res.masks:
                    det["has_mask"] = True
                    # Aquí podrías codificar res.masks[i].xy a JSON si lo necesitas
                
                detections.append(det)
    
    return {"detections": detections}

@app.get("/health")
def health():
    return {"status": "ok"}
