import cv2
import argparse
import os
from ultralytics import YOLO

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--model', type=str, default='./models/bplane_small/weights/bplane_small.engine')
    args = parser.parse_args()

    model = YOLO(args.model)
    source = args.image if args.image else args.video

    if not source:
        print("Error: Please specify either --image or --video parameter.")
        return

    is_image = args.image is not None

    results = model.track(
        source=source,
        conf=0.25,
        iou=0.5,
        imgsz=1024,
        stream=True,
        tracker="botsort.yaml",
        persist=True
    )

    for r in results:
        img = r.orig_img.copy()
        
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            
            ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)

            for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
                label = model.names[cls].upper()
                x1, y1, x2, y2 = map(int, box)
                
                color = (0, 165, 255) if cls == 10 else (0, 255, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                id_text = f"ID:{obj_id} " if obj_id is not None else ""
                info_text = f"{id_text}{label} {conf:.2f}"
                cv2.putText(img, info_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("B-PLANE SMALL INFERENCE", img)
        
        if is_image:
            cv2.waitKey(0)
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()