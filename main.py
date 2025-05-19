from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import uvicorn
import io
import torch

app = FastAPI()

# Load the exported model once when the app starts
model_path = Path("cnn_model_export.pkl")
learn = load_learner(model_path)
learn.model.eval()  # Ensure evaluation mode

@app.get("/")
def read_root():
    return {"message": "FastAI Inference API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image file
        image_bytes = await file.read()
        img = PILImage.create(io.BytesIO(image_bytes))

        # Make prediction
        pred_class, pred_idx, probs = learn.predict(img)

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_class": str(pred_class),
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {
                learn.dls.vocab[i]: float(p) for i, p in enumerate(probs)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

