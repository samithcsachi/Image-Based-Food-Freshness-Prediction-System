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
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc


class CategoryModelEvaluation:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
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
    
    def load_data(self, img_size=(224, 224), batch_size=32):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir / "category" / "test",
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False)
        class_names = test_ds.class_names
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        return test_ds, class_names
    
    def model_eval(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        model = tf.keras.models.load_model(self.model_dir / "category_classifier.keras")

        test_ds, class_names = self.load_data()
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
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