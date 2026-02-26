import pandas as pd
import os
import sys
import glob

AIRCRAFT_CLASSES = ['F16', 'F18', 'F35', 'C130', 'F15', 'J20', 'EF2000', 'RAFALE', 'A10', 'C2', 'AIRCRAFT']

def get_refined_stats(base_dir):
    refined_path = os.path.join(base_dir, "refined_data")
    if not os.path.exists(refined_path):
        print(f"Error: {refined_path} not found.")
        return

    splits = ['train', 'valid', 'test']
    stats = {cls: {'images': 0, 'boxes': 0} for cls in AIRCRAFT_CLASSES}
    
    print(f"{'Split':<8} | {'Class':<10} | {'Images':<8} | {'Boxes':<8}")
    print("-" * 45)

    for split in splits:
        split_img_count = 0
        split_box_count = 0
        
        label_files = glob.glob(os.path.join(refined_path, split, "labels", "*.txt"))
        
        temp_stats = {i: {'images': 0, 'boxes': 0} for i in range(len(AIRCRAFT_CLASSES))}
        
        for lbl in label_files:
            split_img_count += 1
            with open(lbl, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id < len(AIRCRAFT_CLASSES):
                        temp_stats[class_id]['boxes'] += 1
                        split_box_count += 1
            
            # Count unique image per class prefix
            fname = os.path.basename(lbl)
            for i, cls in enumerate(AIRCRAFT_CLASSES):
                if fname.startswith(cls.lower()):
                    temp_stats[i]['images'] += 1

        for i, cls in enumerate(AIRCRAFT_CLASSES):
            if temp_stats[i]['images'] > 0 or temp_stats[i]['boxes'] > 0:
                print(f"{split:<8} | {cls:<10} | {temp_stats[i]['images']:<8} | {temp_stats[i]['boxes']:<8}")
                stats[cls]['images'] += temp_stats[i]['images']
                stats[cls]['boxes'] += temp_stats[i]['boxes']
        print("-" * 45)

    print(f"\n{'TOTAL SUMMARY':<19} | {'Images':<8} | {'Boxes':<8}")
    print("-" * 45)
    for cls in AIRCRAFT_CLASSES:
        print(f"{cls:<19} | {stats[cls]['images']:<8} | {stats[cls]['boxes']:<8}")

def show_raw_stats(base_dir):
    csv_path = os.path.join(base_dir, "raw_data", "positives", "labels_with_split.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['clean_class'] = df['class'].astype(str).str.upper().str.replace('-', '', regex=False).str.strip()
    stats = df.groupby('clean_class')['filename'].nunique().sort_values(ascending=False)

    print(f"{'Aircraft Type':<15} | {'Unique Images':<15}")
    print("-" * 33)
    for aircraft, count in stats.items():
        print(f"{aircraft:<15} | {count:<15}")
    print("-" * 33)
    print(f"Total Unique Classes Found: {len(stats)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['refined', 'refineddata', 'refined_data']:
        get_refined_stats(base_dir)
    else:
        show_raw_stats(base_dir)