import os
from pathlib import Path


project_name = 'Image-Based-Food-Freshness-Prediction-System'


list_of_paths = [
 
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
  
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    "src/detection.py",
    "src/classification.py",
    "src/config.py",
    "src/utils.py",
      
    "artifacts/data/raw/",
    "artifacts/data/processed/",
    "artifacts/models/",
    "artifacts/logs/",

    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_train_classifier.ipynb",
    "notebooks/03_model_evaluation.ipynb",
     
    "requirements.txt",
    "Dockerfile"
              
]

for path_str in list_of_paths:
    p = Path(path_str)

   
    if p.suffix == "" and "." not in p.name and p.name.lower() not in {
        "dockerfile", "license", "readme", "makefile"
    }:
        os.makedirs(p, exist_ok=True)
        print(f"Created directory: {p}")
    else:
     
        os.makedirs(p.parent, exist_ok=True)
        print(f"Created directory: {p.parent}")
        
        if (not p.exists()) or (p.stat().st_size == 0):
            p.touch()
            print(f"Created empty file: {p}")
        else:
            print(f"File already exists: {p} and is not empty.")
