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
import random 

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

    def get_category_image_dir(self, train_split=0.7, val_split=0.15, test_split=0.15):
       
        base_output = Path("artifacts/data/category")
        train_dir = base_output / "train"
        val_dir = base_output / "val"
        test_dir = base_output / "test"

        
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

     
        category_mapping = {
            "Fruits": Config.FRUIT_NAMES,
            "Vegetables": Config.VEGETABLE_NAMES,
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

               
                all_images = []
                for freshness in ["Fresh", "Rotten"]:
                    src_folder = src_cat / freshness
                    if src_folder.exists():
                        for imgfile in src_folder.glob("*.[jp][pn]g"):
                            all_images.append(imgfile)

                if not all_images:
                    self.logger.warning(f"No images found for category: {category}")
                    continue

              
                random.shuffle(all_images)
                total = len(all_images)
                train_end = int(total * train_split)
                val_end = train_end + int(total * val_split)

                splits = {
                    train_dir / category: all_images[:train_end],
                    val_dir / category: all_images[train_end:val_end],
                    test_dir / category: all_images[val_end:]
                }

             
                for dst_cat, img_list in splits.items():
                    dst_cat.mkdir(parents=True, exist_ok=True)
                    for imgfile in img_list:
                        target_path = dst_cat / imgfile.name
                        if not target_path.exists():
                            try:
                                shutil.copy2(imgfile, target_path)
                                if target_path.suffix.lower() == ".png":
                                    self.fix_png_iccp(target_path)
                            except Exception as e:
                                self.logger.error(f"Failed to copy {imgfile}: {e}")

        self.logger.info("Dataset split completed into train/val/test folders.")
        return train_dir, val_dir, test_dir

    def load_data(self, img_size=(224, 224), batch_size=32):
        train_dir, val_dir, test_dir = self.get_category_image_dir()

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
        )

        return train_ds, val_ds, test_ds

    def build_model(self, input_shape=(224, 224, 3), num_classes=10):
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            layers.InputLayer(shape=input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, model, train_ds, val_ds, epochs=30):
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


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

    def save_labels(self, class_names, filename="category_labels.json"):
        save_path = Path(Config.MODEL_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(class_names, f, indent=4)
        self.logger.info(f"Category labels saved to {save_path.resolve()}")

    def model_trainer(self):
        train_ds, val_ds, test_ds = self.load_data()
        categories = Config.FRUIT_NAMES + Config.VEGETABLE_NAMES
        num_classes = len(categories)
        
        model = self.build_model(num_classes=num_classes)
        history = self.train(model, train_ds, val_ds)

        self.save_model(model)
        self.save_metrics(history)
        self.save_labels(train_ds.class_names)
        self.logger.info("Category classifier training complete.")

if __name__ == "__main__":
    cmt = CategoryModelTrainer()
    cmt.model_trainer()