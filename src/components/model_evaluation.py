import os
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir
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

    def save_confusion_matrix(self,y_true, y_pred, class_names, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_roc_curve(self,y_true, y_pred_probs, num_classes, save_path):
        plt.figure(figsize=(8, 6))
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:,1])
            plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc(fpr, tpr))
        else:
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
                plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_classification_report(self,y_true, y_pred, class_names, save_path):
        report = classification_report(y_true, y_pred, target_names=class_names)
        with open(save_path, "w") as f:
            f.write(report)

    def save_error_analysis(self, test_ds, y_true, y_pred, class_names, save_path, n_samples=9):
        wrong_idx = np.where(y_true != y_pred)[0]
        images = []
        for i in wrong_idx[:n_samples]:
            img, _ = test_ds.unbatch().skip(i).take(1).as_numpy_iterator().__next__()
            images.append(img)
        plt.figure(figsize=(10, 10))
        for n, img in enumerate(images):
            plt.subplot(3, 3, n+1)
            plt.imshow(img.astype("uint8"))
            plt.title(f"True: {class_names[y_true[wrong_idx[n]]]}, Pred: {class_names[y_pred[wrong_idx[n]]]}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save_optimization_comparison(self, results_dict, save_path):
        df = pd.DataFrame(results_dict)
        df.to_csv(save_path, index=False)


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

     
        self.save_classification_report(y_true, y_pred, class_names, self.results_dir / "classification_report.txt")
        self.save_confusion_matrix(y_true, y_pred, class_names, self.results_dir / "confusion_matrix.png")
        self.save_roc_curve(y_true, y_pred_probs, num_classes, self.results_dir / "roc_curves.png")
        self.save_error_analysis(test_ds, y_true, y_pred, class_names, self.results_dir / "error_analysis.png")
        results_dict = {
            "stage": ["baseline"],
            "test_accuracy": [test_acc],
            "test_loss": [test_loss]
        }
        self.save_optimization_comparison(results_dict, self.results_dir / "optimization_comparison.csv")

        self.logger.info("Model Evaluation complete.")



if __name__ == "__main__":
    mt = ModelEvaluation()
    mt.model_eval()