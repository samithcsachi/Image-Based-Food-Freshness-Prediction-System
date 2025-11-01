import os 
from src.config import Config
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random


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
