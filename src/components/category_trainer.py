import os
import logging
from pathlib import Path
from src.config import Config
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import json
from PIL import Image
import shutil

class CategoryModelTrainer:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.raw_data_dir = Path(config.RAW_DATA_DIR)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger('CategoryModelTrainer')
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

    def get_category_image_dir(self):
        temp_dir = Path("artifacts/data/category_train")
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

    def load_data(self, img_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
        train_dir = self.get_category_image_dir()
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=val_split,
            subset="training",
            seed=seed
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=val_split,
            subset="validation",
            seed=seed
        )
           

        return train_ds, val_ds

    def build_model(self, input_shape=(224, 224, 3), num_classes=10):

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
       
        model = tf.keras.Sequential([
            layers.InputLayer(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),                   
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, model, train_ds, val_ds, epochs=30):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, min_lr=1e-5)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[early_stopping, reduce_lr]
        )
        return history

    def save_model(self, model, filename="category_classifier.keras"):
        save_path = Path(Config.MODEL_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)  
        model.save(save_path)
        self.logger.info(f"Model saved to {save_path.resolve()}")

    def save_metrics(self, history, filename="category_history.json"):
        save_path = Path(Config.MODEL_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Training metrics saved to {save_path.resolve()}")


    def model_trainer(self):
        train_ds, val_ds = self.load_data()
        categories = Config.FRUIT_NAMES + Config.VEGETABLE_NAMES
        num_classes = len(categories)
        model = self.build_model(num_classes=num_classes)
        history = self.train(model, train_ds, val_ds)
        self.save_model(model)
        self.save_metrics(history)
        self.logger.info("Category classifier training complete.")

if __name__ == "__main__":
    cmt = CategoryModelTrainer()
    cmt.model_trainer()
