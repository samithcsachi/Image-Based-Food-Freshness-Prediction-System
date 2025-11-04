import os
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir
from PIL import Image
import tensorflow as tf 



class ModelTrainer:
    def __init__(self, config: Config = Config()):
        
        self.config = config
        self.processed_data_dir = Path(config.PROCESSED_DATA_DIR)
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        return logger
    
    def load_data(self):

        data = tf.keras.utils.image_dataset_from_directory('processed')
