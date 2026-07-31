import cv2
from ultralytics import YOLO

def main():
    model = YOLO("models/exp.onnx")
    img = cv2.imread("test.jpg")
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    for conf in thresholds:
        results = model(img, conf=conf, verbose=False)
        boxes = results[0].boxes
        if boxes is not None:
            num_dets = len(boxes)
            confs = boxes.conf.cpu().numpy()
            avg_conf = confs.mean() if num_dets > 0 else 0.0
            print(f"Conf Threshold: {conf:.2f} -> Detections: {num_dets}, Avg Conf: {avg_conf:.4f}")
            if conf == 0.25:
                print("Boxes for conf 0.25:")
                for i in range(num_dets):
                    c = confs[i]
                    cls = int(boxes.cls[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    print(f"  Cls: {cls}, Conf: {c:.4f}, Box: {xyxy}")
        else:
            print(f"Conf Threshold: {conf:.2f} -> Detections: 0")

if __name__ == "__main__":
    main()
