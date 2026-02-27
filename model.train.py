# Copyright 2026 Bulut Müftüoğlu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

