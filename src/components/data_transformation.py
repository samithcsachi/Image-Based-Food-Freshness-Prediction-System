import os
import json
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir
import random
from PIL import Image



class DataTransformation:
    def __init__(self, config: Config = Config()):
        
        self.config = config
        self.raw_data_dir = Path(config.RAW_DATA_DIR)
        self.processed_data_dir = Path(config.PROCESSED_DATA_DIR)
        self.img_size = (224, 224)  
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('DataTransformation')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        return logger
    

    def train_test_split(self,image_paths_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        splits = {'train': [], 'val': [], 'test': []}
        for cls, paths in image_paths_dict.items():
            paths = list(paths)  
            random.shuffle(paths)
            n_total = len(paths)
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)
            splits['train'].extend([(path, cls) for path in paths[:n_train]])
            splits['val'].extend([(path, cls) for path in paths[n_train:n_train+n_val]])
            splits['test'].extend([(path, cls) for path in paths[n_train+n_val:]])
        random.shuffle(splits['train'])
        random.shuffle(splits['val'])
        random.shuffle(splits['test'])
        return splits
    

    def preprocess_and_save_split(self, split_list, split_type):
        
        for img_path, label in split_list:
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.img_size)
           
            dest_dir = self.processed_data_dir / split_type / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / img_path.name
            img.save(dest_path)
        self.logger.info(f"Saved {len(split_list)} images to {split_type} set.")

    def transform_dataset_pipeline(self):
        self.logger.info("Starting dataset transformation pipeline")
        
  
        image_paths_dict = raw_img_dir(self.raw_data_dir, self.config.CATEGORY_NAMES)


        splits = self.train_test_split(image_paths_dict)

 
        for split_type in ['train', 'val', 'test']:
            split_list = splits[split_type]
            self.preprocess_and_save_split(split_list, split_type)

        self.logger.info("Transformation pipeline completed.")

if __name__ == "__main__":
    dt = DataTransformation()
    dt.transform_dataset_pipeline()
