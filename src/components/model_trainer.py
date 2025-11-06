import os
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir
from PIL import Image
import tensorflow as tf 
from keras.callbacks import EarlyStopping
import json




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
    
    def load_data(self, img_size=(224, 224), batch_size=32):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.processed_data_dir / "train",
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.processed_data_dir / "val",
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.processed_data_dir / "test",
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        return train_ds, val_ds, test_ds
    
    def build_model(self, input_shape=(224, 224, 3), num_classes=2):
    
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False
        )
        base_model.trainable = False  

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )
        return model
    
    def train(self, model, train_ds, val_ds, epochs=50):
        
        early_stopping = EarlyStopping(
            monitor='val_loss',    
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping]  
        )
    
        return history
    
    def save_model(self, model, save_path = os.path.join('..', 'models',"mobilenetv2_baseline.keras")):
        model.save(self.processed_data_dir.parent / save_path)
        self.logger.info(f"Model saved to {self.processed_data_dir.parent / save_path}")



    def save_metrics(self, history, save_path = os.path.join('..', 'models',"baseline_history.json")):
        
        metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(self.processed_data_dir.parent / save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Training metrics saved to {self.processed_data_dir.parent / save_path}")


    def model_trainer(self):
    
        train_ds, val_ds, test_ds = self.load_data()
    
        model = self.build_model(num_classes=2)
        
        history = self.train(model, train_ds, val_ds)
        
        self.save_model(model)
        self.save_metrics(history)
        self.logger.info("Baseline model training complete.")


if __name__ == "__main__":
        mt = ModelTrainer()
        mt.model_trainer()
