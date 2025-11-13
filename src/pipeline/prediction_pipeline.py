import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import json

from src.config import Config

CATEGORY_MODEL_PATH = Path(Config.MODEL_DIR) / "category_classifier.keras"
FRESHNESS_MODEL_PATH = Path(Config.MODEL_DIR) / "mobilenetv2_baseline.keras"

CATEGORY_LABELS_PATH = Path(Config.MODEL_DIR) / "category_labels.json"
with open(CATEGORY_LABELS_PATH, "r") as f:
    CATEGORY_LABELS = json.load(f)

FRESHNESS_LABELS = ["Fresh", "Rotten"]

CATEGORY_IMG_SIZE = (224, 224)
FRESHNESS_IMG_SIZE = (224, 224)

class PredictionPipeline:
    def __init__(self):
        self.category_model = tf.keras.models.load_model(CATEGORY_MODEL_PATH)
        self.freshness_model = tf.keras.models.load_model(FRESHNESS_MODEL_PATH)
        self.category_labels = CATEGORY_LABELS
        self.freshness_labels = FRESHNESS_LABELS

    def _preprocess_image(self, img, target_size, normalize=True):
        if isinstance(img, (str, Path)):
            img_path = Path(img)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path.resolve()}")
            img = Image.open(str(img_path)).convert("RGB")
        elif isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                img = Image.fromarray(img)
            else:
                raise ValueError("NumPy input must be shape (H, W, 3)")
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized).astype("float32")
        if normalize:
            img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img

 
    def predict(self, img):
   
        cat_img_array, pil_img = self._preprocess_image(img, CATEGORY_IMG_SIZE, normalize=True)
        cat_pred = self.category_model.predict(cat_img_array)
        cat_idx = int(np.argmax(cat_pred))
        cat_label = self.category_labels[cat_idx]
        cat_score = float(np.max(cat_pred))

      
        fresh_img_array, _ = self._preprocess_image(img, FRESHNESS_IMG_SIZE, normalize=True)
        fresh_pred = self.freshness_model.predict(fresh_img_array)
        fresh_idx = int(np.argmax(fresh_pred))
        fresh_label = self.freshness_labels[fresh_idx]
        fresh_score = float(np.max(fresh_pred))

        return {
            "category": {"label": cat_label, "idx": cat_idx, "score": cat_score},
            "freshness": {"label": fresh_label, "idx": fresh_idx, "score": fresh_score},
            "pil_img": pil_img
        }

    def annotate(self, img, result, font_size=28):
        pil_img = result["pil_img"]
        draw = ImageDraw.Draw(pil_img)
        text = f"{result['category']['label']} ({result['category']['score']:.2f}) | " \
               f"{result['freshness']['label']} ({result['freshness']['score']:.2f})"
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        draw.rectangle([0, 0, pil_img.width, font_size+8], fill=(0,0,0,160))
        draw.text((5, 2), text, fill=(255, 255, 255), font=font)
        return pil_img

if __name__ == "__main__":
    pipeline = PredictionPipeline()

    img_path = r"artifacts\data\category\test\Apple\rottenApple (175).jpg"
    
    result = pipeline.predict(img_path)
    
    print("Prediction:", result["category"], "|", result["freshness"])

    annotated_img = pipeline.annotate(img_path, result)

    annotated_img.save(r"artifacts\results\annotated_result.jpg")