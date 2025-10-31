import os
import json
import logging
from pathlib import Path
from src.config import Config
from src.utils import raw_img_dir
from PIL import Image
from datetime import datetime
from collections import defaultdict


class DataIngestion:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.raw_data_dir = Path(config.RAW_DATA_DIR)
        self.metadata_file = Path(config.METADATA_FILE)
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('DataIngestion')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(ch)
        return logger
    
       
    def validate_data(self):
        self.logger.info("Starting data validation")
        valid = True
        validation_report = {'classes': {'Fresh': {}, 'Rotten': {}}, 'total_images': 0}

       
        image_paths = raw_img_dir(self.raw_data_dir, self.config.CATEGORY_NAMES)

        total_images = 0
        for cls in self.config.CLASS_NAMES:
            count = len(image_paths[cls])
            total_images += count
            if count == 0:
                valid = False
                self.logger.error(f"No images found for class '{cls}'")
                validation_report['classes'][cls] = {
                    'status': 'error',
                    'message': 'No images found',
                    'image_count': 0
                }
            else:
                self.logger.info(f"Class '{cls}' contains {count} images")
                validation_report['classes'][cls] = {
                    'status': 'success',
                    'message': f'{count} images found',
                    'image_count': count
                }
        validation_report['total_images'] = total_images
        validation_report['validation_passed'] = valid

        report_path = self.raw_data_dir.parent / 'validation_report.json'
        os.makedirs(report_path.parent, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=4)

        self.logger.info(f"Validation report saved to {report_path}")

        if valid:
            self.logger.info(f"Data validation PASSED. Total images: {total_images}")
        else:
            self.logger.error("Data validation FAILED. Check validation report for details.")

        return valid, validation_report

    def check_image_integrity(self):
        self.logger.info("Checking image file integrity")
        corrupted_files = []

        
        image_paths = raw_img_dir(self.raw_data_dir, self.config.CATEGORY_NAMES)
        for cls, files in image_paths.items():
            for img_path in files:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted_files.append(str(img_path))
                    self.logger.warning(f"Corrupted image found: {img_path} - {str(e)}")

        if corrupted_files:
            self.logger.warning(f"Found {len(corrupted_files)} corrupted image files")
        else:
            self.logger.info("All image files passed integrity check")
        return corrupted_files

    def generate_metadata(self):
        self.logger.info("Generating dataset metadata")
        metadata = {
            'dataset_info': {
                'name': 'Fruits and Vegetables Dataset',
                'classes': self.config.CLASS_NAMES,
                'task_type': 'binary_classification',
                'ingestion_date': datetime.now().isoformat()
            },
            'class_distribution': {},
            'total_images': 0
        }

        
        image_paths = raw_img_dir(self.raw_data_dir, self.config.CATEGORY_NAMES)
        for cls in self.config.CLASS_NAMES:
            count = len(image_paths[cls])
            metadata['class_distribution'][cls] = count
            metadata['total_images'] += count

        if metadata['total_images'] > 0:
            for cls in self.config.CLASS_NAMES:
                percentage = (metadata['class_distribution'][cls] / metadata['total_images']) * 100
                metadata['class_distribution'][f'{cls}_percentage'] = round(percentage, 2)

        os.makedirs(self.metadata_file.parent, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Metadata saved to {self.metadata_file}")
        self.logger.info(f"Dataset summary: {metadata['class_distribution']}")

        return metadata

    def initiate_data_ingestion(self, check_integrity: bool = False):
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING DATA INGESTION PIPELINE")
            self.logger.info("=" * 60)

           
            is_valid, validation_report = self.validate_data()
            if not is_valid:
                raise Exception("Data validation failed. Please check the validation report.")

            corrupted_files = []
            if check_integrity:
                corrupted_files = self.check_image_integrity()

            metadata = self.generate_metadata()

            ingestion_results = {
                "status": "success",
                "raw_data_dir": str(self.raw_data_dir),
                "metadata_file": str(self.metadata_file),
                "total_images": metadata['total_images'],
                "class_distribution": metadata['class_distribution'],
                "corrupted_files": corrupted_files,
                "validation_passed": is_valid
            }

            self.logger.info("=" * 60)
            self.logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total images ingested: {metadata['total_images']}")
            self.logger.info(f"Fresh images: {metadata['class_distribution'].get('Fresh', 0)}")
            self.logger.info(f"Rotten images: {metadata['class_distribution'].get('Rotten', 0)}")
            self.logger.info("=" * 60)

            return ingestion_results

        except Exception as e:
            self.logger.error(f"Data ingestion failed: {str(e)}")
            raise

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    results = data_ingestion.initiate_data_ingestion(        
        check_integrity=True
    )
    print("\nIngestion Results:")
    print(f"Status: {results['status']}")
    print(f"Total images: {results['total_images']}")
    print(f"Fresh: {results['class_distribution']['Fresh']}")
    print(f"Rotten: {results['class_distribution']['Rotten']}")