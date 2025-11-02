import os 
from src.config import Config
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import numpy as np 


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



