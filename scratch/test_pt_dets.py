import cv2
from ultralytics import YOLO

def main():
    model = YOLO("models/exp.pt")
    cap = cv2.VideoCapture("test.mp4")
    
    ret, frame = cap.read()
    if not ret:
        return
            
    print("Testing exp.pt on first frame...")
    results = model(frame, conf=0.01, verbose=False)
    boxes = results[0].boxes
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            c = box.conf.item()
            cls = int(box.cls.item())
            print(f"exp.pt -> Cls: {cls}, Conf: {c:.4f}")
    else:
        print("exp.pt -> No detections.")

if __name__ == "__main__":
    main()
