import os
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir,save_classification_report, save_confusion_matrix, save_error_analysis, save_optimization_comparison, save_roc_curve
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc


class ModelEvaluation:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.processed_data_dir = Path(config.PROCESSED_DATA_DIR)
        self.results_dir = Path(config.RESULTS_DIR)      
        self.model_dir = Path(config.MODEL_DIR)         
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('Model Evaluation')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        return logger

    def load_data(self, img_size=(224, 224), batch_size=32):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.processed_data_dir / "test",
            label_mode="int",  
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        return test_ds

    def model_eval(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        model = tf.keras.models.load_model(self.model_dir / "mobilenetv2_baseline.keras")

        test_ds = self.load_data()
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        class_names = test_ds.class_names
        num_classes = len(class_names)

     
        save_classification_report(y_true, y_pred, class_names, self.results_dir / "classification_report.txt")
        save_confusion_matrix(y_true, y_pred, class_names, self.results_dir / "confusion_matrix.png")
        save_roc_curve(y_true, y_pred_probs, num_classes, self.results_dir / "roc_curves.png")
        save_error_analysis(test_ds, y_true, y_pred, class_names, self.results_dir / "error_analysis.png")
        results_dict = {
            "stage": ["baseline"],
            "test_accuracy": [test_acc],
            "test_loss": [test_loss]
        }
        save_optimization_comparison(results_dict, self.results_dir / "optimization_comparison.csv")

        self.logger.info("Model Evaluation complete.")



if __name__ == "__main__":
    mt = ModelEvaluation()
    mt.model_eval()