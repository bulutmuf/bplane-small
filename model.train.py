import os
import yaml
import random
import torch
import numpy as np
from ultralytics import YOLO

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_hyp(hyp_path: str) -> dict:
    with open(hyp_path, "r") as f:
        return yaml.safe_load(f)

def train():
    set_seed(42)

    data_yaml_path = os.path.abspath("./config/data.yaml")
    hyp_yaml_path = os.path.abspath("./config/hyp.yaml")
    project_dir = os.path.abspath("./models")
    model_name = "bplane_small" 
    
    hyp = load_hyp(hyp_yaml_path)
    print(f"Loaded hyperparameters from {hyp_yaml_path}")

  
    model = YOLO("yolo11s.pt") 

    model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=1024,
        batch=64,
        device=0,
        workers=16,
        cache=True,
        project=project_dir,
        name=model_name,
        exist_ok=True,
        patience=20,
        cos_lr=True,
        seed=42,
        deterministic=True,
        val=True,
        save=True,
        amp=True, 
        **hyp 
    )

    print("Pipeline completed successfully.")

if __name__ == "__main__":

    train()
