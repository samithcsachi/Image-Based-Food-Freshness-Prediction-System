import os 


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'processed')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'results')

    METADATA_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'data', 'metadata.json')
    CATEGORY_NAMES =  ['Fruits','Vegetables']
    FRUIT_NAMES = ['Apple', 'Banana', 'Mango', 'Orange', 'Strawberry']
    VEGETABLE_NAMES = ['Bellpepper', 'Carrot', 'Cucumber', 'Potato', 'Tomato' ]
    CLASS_NAMES = ['Fresh', 'Rotten']
  