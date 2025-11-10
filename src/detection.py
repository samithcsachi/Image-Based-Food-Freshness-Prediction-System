import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from src.config import Config

class YOLODetector:
    def __init__(self):
        self.model_path = Path(Config.MODEL_DIR) / "yolov8n.pt"
        self.results_dir = Path(Config.RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_results_dir = self.results_dir / "yolo_detections"
        self.yolo_results_dir.mkdir(parents=True, exist_ok=True)
        self.yolo = YOLO(str(self.model_path))
        self.food_class_names = Config.FRUIT_NAMES + Config.VEGETABLE_NAMES


    def detect(self, image_path):
        results = self.yolo(image_path)
        img_results = []
        for result in results:
            boxes = []
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = self.yolo.names[class_id]
                if class_name in self.food_class_names:
                    boxes.append({
                        "box": box.xyxy[0].cpu().numpy().tolist(),
                        "score": float(box.conf),
                        "label": class_name
                    })
            img_results.append({
                "image_path": image_path,
                "boxes": boxes
            })
        return img_results

    def visualize_and_save(self, image_path, results):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
  
        save_path = self.yolo_results_dir / f"{base_filename}_yolo_detection.png"
        yolo_result = self.yolo(image_path)[0]
        annotated_img = yolo_result.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_img)
        plt.axis('off')
        plt.title('YOLOv8 Detection')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved detection to {save_path}")


    def detect_and_visualize(self, image_path):
        results = self.detect(image_path)
        self.visualize_and_save(image_path, results)
        return results

if __name__ == "__main__":
    detector = YOLODetector()
   
    processed_test_fresh = Path(Config.PROCESSED_DATA_DIR) / "test" / "Fresh"
    processed_test_rotten = Path(Config.PROCESSED_DATA_DIR) / "test" / "Rotten"


    all_image_files = list(processed_test_fresh.rglob("*.[jp][pn]g")) + list(processed_test_rotten.rglob("*.[jp][pn]g"))
    print(f"Found {len(all_image_files)} processed test images.")
    for img_file in all_image_files:
        print(f"Processing {img_file}...")
        results = detector.detect_and_visualize(str(img_file))
