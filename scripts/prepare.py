import cv2
import os
import sys
import pandas as pd
import shutil
from tqdm import tqdm

# Master ID sözlüğümüz (data.yaml ile birebir uyumlu)
AIRCRAFT_MAP = {
    'F16': 0, 
    'F18': 1, 
    'F35': 2, 
    'C130': 3, 
    'F15': 4,
    'J20': 5, 
    'EF2000': 6, 
    'RAFALE': 7, 
    'A10': 8, 
    'C2': 9
}
FALLBACK_ID = 10

def get_class_id(cls_name):
    """
    Sınıf ismini alır, listede varsa karşılık gelen ID'yi, 
    yoksa 10 (other aircraft) ID'sini döner.
    """
    name = str(cls_name).strip().upper()
    name = name.replace("-", "") # F-16 veya F16 yazım farklarını eşitlemek için
    
    for key, cid in AIRCRAFT_MAP.items():
        if key in name:
            return cid
    return FALLBACK_ID

def convert_to_yolo(real_w, real_h, xmin, ymin, xmax, ymax):
    box_w = xmax - xmin
    box_h = ymax - ymin
    center_x = xmin + (box_w / 2)
    center_y = ymin + (box_h / 2)
    
    return (
        center_x / real_w,
        center_y / real_h,
        box_w / real_w,
        box_h / real_h
    )

def process_model(target_model):
    target_model_upper = target_model.upper()
    
    if target_model_upper not in AIRCRAFT_MAP:
        print(f"Hata: {target_model_upper} ana ucak listesinde (AIRCRAFT_MAP) bulunamadi.")
        print(f"Gecerli ucaklar: {list(AIRCRAFT_MAP.keys())}")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    raw_pos_dir = os.path.join(base_dir, "raw_data", "positives")
    img_src_dir = os.path.join(raw_pos_dir, "dataset")
    csv_path = os.path.join(raw_pos_dir, "labels_with_split.csv")
    refined_dir = os.path.join(base_dir, "refined_data")
    
    print(f"CSV Yükleniyor: {csv_path}...")
    df_master = pd.read_csv(csv_path)
    
    # Hedef uçağı filtreleme (Büyük/küçük harf veya tire duyarsız)
    mask = df_master['class'].astype(str).str.upper().str.replace('-', '').str.contains(target_model_upper)
    target_filenames = df_master[mask]['filename'].unique()
    
    total_found = len(target_filenames)
    print(f"Toplam benzersiz {target_model_upper} gorseli bulundu: {total_found}")

    if total_found == 0:
        print("Uyarı: Bu ucak icin hic gorsel bulunamadi. Islem iptal ediliyor.")
        return

    if total_found >= 1000:
        counts = {'train': 700, 'valid': 200, 'test': 100}
    else:
        counts = {
            'train': int(total_found * 0.7),
            'valid': int(total_found * 0.2),
            'test': total_found - (int(total_found * 0.7) + int(total_found * 0.2))
        }
    
    print(f"Dagitim: {counts}")

    start_idx = 0
    for split, count in counts.items():
        if count == 0: continue
        print(f"\nProcessing {split.upper()} set ({count} files)...")
        
        img_dest = os.path.join(refined_dir, split, "images")
        lbl_dest = os.path.join(refined_dir, split, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        selected_files = target_filenames[start_idx : start_idx + count]
        
        for i, fname in enumerate(tqdm(selected_files)):
            # Dosya isimlendirmesi (örn: f15_0001.jpg)
            new_name = f"{target_model.lower()}_{start_idx + i + 1:04d}"
            src_img_path = os.path.join(img_src_dir, f"{fname}.jpg")
            
            if not os.path.exists(src_img_path):
                continue
                
            img = cv2.imread(src_img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            shutil.copy(src_img_path, os.path.join(img_dest, f"{new_name}.jpg"))

            img_labels = df_master[df_master['filename'] == fname]
            with open(os.path.join(lbl_dest, f"{new_name}.txt"), "w") as f:
                for _, row in img_labels.iterrows():
                    # AKILLI ETİKETLEME SİSTEMİ DEVREDE
                    cid = get_class_id(row['class'])
                    
                    yolo = convert_to_yolo(w, h, row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                    yolo = [max(0.0, min(1.0, val)) for val in yolo]
                    
                    f.write(f"{cid} {' '.join([f'{x:.6f}' for x in yolo])}\n")
        
        start_idx += count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim hatasi! Gecerli kullanim: python prepare.py <ucak_modeli>")
        print("Ornek: python prepare.py f15")
        sys.exit(1)
        
    target = sys.argv[1]
    process_model(target)