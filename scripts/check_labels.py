import cv2
import os
import sys
import random
import glob

AIRCRAFT_NAMES = ['f16', 'f18', 'f35', 'c130', 'f15', 'j20', 'ef2000', 'rafale', 'a10', 'c2', 'aircraft']

def verify_label_alignment(model_name, img_id, split):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    refined_path = os.path.join(base_dir, "refined_data")

    if model_name.lower() == "random" and img_id == "0001":
        all_images = glob.glob(os.path.join(refined_path, "*", "images", "*.jpg"))
        if not all_images: print("No images found!"); return
        img_path = random.choice(all_images)
        lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        split = img_path.split(os.sep)[-3]
    elif img_id.lower() == "random":
        all_images = glob.glob(os.path.join(refined_path, "*", "images", f"{model_name.lower()}_*.jpg"))
        if not all_images: print(f"No images found for {model_name}!"); return
        img_path = random.choice(all_images)
        lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        split = img_path.split(os.sep)[-3]
    else:
        img_id = str(img_id).zfill(4)
        img_filename = f"{model_name.lower()}_{img_id}"
        img_path = os.path.join(refined_path, split, "images", f"{img_filename}.jpg")
        lbl_path = os.path.join(refined_path, split, "labels", f"{img_filename}.txt")

    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        print(f"File not found: {img_path}"); return

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        cid = int(parts[0])
        name = AIRCRAFT_NAMES[cid].upper() if cid < len(AIRCRAFT_NAMES) else "UNKNOWN"
        cx, cy, bw, bh = map(float, parts[1:])
        
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{name} (ID:{cid})", (x1, max(y1 - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out = os.path.join(base_dir, "label_verification.jpg")
    cv2.imwrite(out, img)
    print(f"Verified {os.path.basename(img_path)} in {split}. Exported to: {out}")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "f16"
    i = sys.argv[2] if len(sys.argv) > 2 else "0001"
    s = sys.argv[3] if len(sys.argv) > 3 else "train"
    verify_label_alignment(m, i, s)