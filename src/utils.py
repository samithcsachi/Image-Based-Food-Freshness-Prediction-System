import os 
from src.config import Config
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc



def raw_img_dir(raw_data_dir, category_names):
    image_paths = {'Fresh': [], 'Rotten': []}
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    for category in category_names:
        category_path = Path(raw_data_dir) / category
        if not category_path.exists():
            continue
        for item_folder in category_path.iterdir():
            if not item_folder.is_dir():
                continue
            for freshness_folder in item_folder.iterdir():
                if not freshness_folder.is_dir():
                    continue
                label = freshness_folder.name.strip().capitalize()  
                if label in image_paths:
                    for ext in image_extensions:
                        image_paths[label].extend(freshness_folder.glob(ext))
                        image_paths[label].extend(freshness_folder.glob(ext.upper()))
    return image_paths




def plot_class_distribution(class_counts, title="Class Distribution"):
    
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Image Count")
    plt.xlabel("Class")
    plt.show()

def show_random_images(image_paths_dict, num_per_class=5, figsize=(12, 4)):
    
    for cls, paths in image_paths_dict.items():
        sample_paths = random.sample(paths, min(num_per_class, len(paths)))
        plt.figure(figsize=figsize)
        for i, img_path in enumerate(sample_paths):
            plt.subplot(1, len(sample_paths), i+1)
            plt.imshow(Image.open(img_path))
            plt.axis("off")
            plt.title(f"{cls}")
        plt.suptitle(f"Sample Images: {cls}")
        plt.show()

def show_image_by_path(path):
   
    img = Image.open(path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(str(path))
    plt.show()


def plot_image_sizes(image_paths_dict):
    widths, heights = [], []

    for cls, paths in image_paths_dict.items():
        for p in paths:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)

    plt.figure(figsize=(8,6))
    plt.scatter(widths, heights, alpha=0.3)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Resolution Distribution")
    plt.show()

    plt.figure(figsize=(8,4))
    plt.hist([w/h for w,h in zip(widths,heights)], bins=30)
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Aspect Ratio (W/H)")
    plt.show()



def plot_color_histogram(img_path):
    img = np.array(Image.open(img_path))

    colors = ('r','g','b')
    plt.figure(figsize=(8,4))

    for i, col in enumerate(colors):
        plt.hist(img[:,:,i].ravel(), bins=256, alpha=0.5, label=f'{col} channel')

    plt.legend()
    plt.title(f"Color Histogram: {img_path}")
    plt.show()


def compute_blur_score(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

def plot_blur_distribution(image_paths_dict):
    blur_scores = []

    for cls, paths in image_paths_dict.items():
        for p in paths:
            blur_scores.append(compute_blur_score(p))

    plt.figure(figsize=(8,4))
    plt.hist(blur_scores, bins=40)
    plt.title("Blur Score Distribution (Laplacian Variance)")
    plt.xlabel("Sharpness Score")
    plt.ylabel("Count")
    plt.show()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
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

def save_roc_curve(y_true, y_pred_probs, num_classes, save_path):
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

def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path, "w") as f:
        f.write(report)

def save_error_analysis(test_ds, y_true, y_pred, class_names, save_path, n_samples=9):
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

def save_optimization_comparison(results_dict, save_path):
    df = pd.DataFrame(results_dict)
    df.to_csv(save_path, index=False)




