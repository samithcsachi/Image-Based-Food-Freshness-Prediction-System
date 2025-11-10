import os
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir,save_classification_report, save_confusion_matrix, save_error_analysis, save_optimization_comparison, save_roc_curve
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import shutil


class CategoryModelEvaluation:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.raw_data_dir = Path(config.RAW_DATA_DIR)  
        self.results_dir = Path(config.RESULTS_DIR)
        self.model_dir = Path(config.MODEL_DIR)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('Category Model Evaluation')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        return logger
    


    def fix_png_iccp(self, path):
        try:
            img = Image.open(path)
            img.save(path, icc_profile=None)
        except:
            pass
    
    def get_category_test_image_dir(self):

        temp_dir = Path("artifacts/data/category_test")
        temp_dir.mkdir(parents=True, exist_ok=True)

        category_mapping = {
            "Fruits": Config.FRUIT_NAMES,
            "Vegetables": Config.VEGETABLE_NAMES
        }

        for main_cat, categories in category_mapping.items():
            base = self.raw_data_dir / main_cat
            if not base.exists():
                self.logger.warning(f"Main category not found: {base}")
                continue

            for category in categories:
                src_cat = base / category
                if not src_cat.exists():
                    self.logger.warning(f"Category not found: {src_cat}")
                    continue

                dst_cat = temp_dir / category
                dst_cat.mkdir(exist_ok=True)

                for freshness in ["Fresh", "Rotten"]:
                    src_folder = src_cat / freshness
                    if src_folder.exists():
                        for imgfile in src_folder.glob("*.[jp][pn]g"):
                            target_path = dst_cat / imgfile.name
                            if not target_path.exists():
                                try:
                                    shutil.copy2(imgfile, target_path)
                                    if target_path.suffix.lower() == ".png":
                                        self.fix_png_iccp(target_path)
                                except Exception as e:
                                    self.logger.error(f"Failed to copy {imgfile}: {e}")
        
        
        created_categories = [d.name for d in temp_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Created categories: {sorted(created_categories)}")
        return temp_dir

    def load_data(self, img_size=(224, 224), batch_size=32):
        test_dir = self.get_category_test_image_dir()
        
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False)
          
        
        return test_ds

    
    def model_eval(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        model = tf.keras.models.load_model(self.model_dir / "category_classifier.keras")

        test_ds = self.load_data()
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        class_names = test_ds.class_names
        num_classes = len(class_names)

        save_classification_report(y_true, y_pred, class_names, self.results_dir / "category_classification_report.txt")
        save_confusion_matrix(y_true, y_pred, class_names, self.results_dir / "category_confusion_matrix.png")
        save_roc_curve(y_true, y_pred_probs, num_classes,self.results_dir / "category_roc_curves.png")
        save_error_analysis(test_ds, y_true, y_pred, class_names, self.results_dir / "category_error_analysis.png")
        results_dict = {
            "stage": ["category_classifier"],
            "test_accuracy": [test_acc],
            "test_loss": [test_loss]
        }
        save_optimization_comparison(results_dict, self.results_dir / "category_optimization_comparison.csv")

        self.logger.info("Category Classifier Model Evaluation complete.")

if __name__ == "__main__":
    cme = CategoryModelEvaluation()
    cme.model_eval()
