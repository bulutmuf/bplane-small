import cv2
import os
import pandas as pd
import shutil
import random
from tqdm import tqdm

AIRCRAFT_MAP = ['F16', 'F18', 'F35', 'C130', 'F15', 'J20', 'EF2000', 'RAFALE', 'A10', 'C2']
TARGET_TOTAL = 1000
AIRCRAFT_ID = 10

def process_general_aircraft():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    raw_pos_dir = os.path.join(base_dir, "raw_data", "positives")
    img_src_dir = os.path.join(raw_pos_dir, "dataset")
    csv_path = os.path.join(raw_pos_dir, "labels_with_split.csv")
    refined_dir = os.path.join(base_dir, "refined_data")
    
    df = pd.read_csv(csv_path)
    df['clean_class'] = df['class'].astype(str).str.upper().str.replace('-', '', regex=False).str.strip()

    # Ana 10 uçağı filtrele
    others_df = df[~df['clean_class'].isin(AIRCRAFT_MAP)]
    other_classes = others_df['clean_class'].unique()
    
    selected_filenames = []
    
    # Her sınıftan dengeli örnek al (Çeşitlilik için)
    per_class_limit = 15 
    for cls in other_classes:
        cls_files = others_df[others_df['clean_class'] == cls]['filename'].unique().tolist()
        random.shuffle(cls_files)
        selected_filenames.extend(cls_files[:per_class_limit])

    # Karıştır ve ihtiyacımız kadarını (1000) al
    random.shuffle(selected_filenames)
    selected_filenames = selected_filenames[:TARGET_TOTAL]
    
    print(f"Total unique 'Other Aircraft' selected: {len(selected_filenames)}")
    
    counts = {'train': 700, 'valid': 200, 'test': 100}
    start_idx = 0

    for split, count in counts.items():
        print(f"\nProcessing {split.upper()} set...")
        img_dest = os.path.join(refined_dir, split, "images")
        lbl_dest = os.path.join(refined_dir, split, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        current_files = selected_filenames[start_idx : start_idx + count]
        
        for i, fname in enumerate(tqdm(current_files)):
            new_name = f"aircraft_{start_idx + i + 1:04d}"
            src_path = os.path.join(img_src_dir, f"{fname}.jpg")
            
            if not os.path.exists(src_path): continue
            
            img = cv2.imread(src_path)
            if img is None: continue
            h, w, _ = img.shape
            
            # Resmi kopyala
            shutil.copy(src_path, os.path.join(img_dest, f"{new_name}.jpg"))
            
            # Etiketleri oluştur (Bu resimdeki TÜM uçakları ID 10 olarak işaretle)
            # Not: Bu basit versiyonda 'diğer' klasöründeki her şeyi 10 kabul ediyoruz.
            img_labels = df[df['filename'] == fname]
            with open(os.path.join(lbl_dest, f"{new_name}.txt"), "w") as f:
                for _, row in img_labels.iterrows():
                    # YOLO formatına çevir
                    box_w = row['xmax'] - row['xmin']
                    box_h = row['ymax'] - row['ymin']
                    cx = (row['xmin'] + (box_w / 2)) / w
                    cy = (row['ymin'] + (box_h / 2)) / h
                    bw = box_w / w
                    bh = box_h / h
                    f.write(f"{AIRCRAFT_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        start_idx += count

if __name__ == "__main__":
    process_general_aircraft()