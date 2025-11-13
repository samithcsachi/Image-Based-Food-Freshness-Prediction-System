from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI(title="Food Freshness Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pipeline = PredictionPipeline()

def read_imagefile(file) -> np.ndarray:
    img = Image.open(file)
    return np.array(img)

@app.post("/predict/image")
async def predict_image(image: UploadFile = File(...)):
    img = read_imagefile(image.file)
    result = pipeline.predict(img)
    annotated_img = pipeline.annotate(img, result)
    out_path = "annotated_result.jpg"
    annotated_img.save(out_path)
    return {
        "category": result["category"],
        "freshness": result["freshness"],
        "annotated_filepath": out_path
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
