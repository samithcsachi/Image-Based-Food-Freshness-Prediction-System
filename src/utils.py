import os 
from src.config import Config
from pathlib import Path

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