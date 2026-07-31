import cv2
from ultralytics import YOLO

def main():
    image_path = r"C:\Users\Mahnoor Shabbir\Downloads\img1211_jpg.rf.56faddade9c4ea80f3bc86a5cbe4d4cf.jpg"
    img = cv2.imread(image_path)
    
    if img is None:
        print("Failed to load image from:", image_path)
        return
        
    print("Testing exp.pt on user validation image...")
    model_pt = YOLO("models/exp.pt")
    results_pt = model_pt(img, conf=0.01, verbose=False)
    boxes_pt = results_pt[0].boxes
    if boxes_pt is not None and len(boxes_pt) > 0:
        for box in boxes_pt:
            print(f"exp.pt -> Cls: {int(box.cls.item())}, Conf: {box.conf.item():.4f}")
    else:
        print("exp.pt -> No detections.")

if __name__ == "__main__":
    main()
