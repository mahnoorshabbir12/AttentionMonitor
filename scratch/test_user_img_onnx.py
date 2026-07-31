import cv2
from ultralytics import YOLO

def main():
    image_path = r"C:\Users\Mahnoor Shabbir\Downloads\img1211_jpg.rf.56faddade9c4ea80f3bc86a5cbe4d4cf.jpg"
    img = cv2.imread(image_path)
    
    if img is None:
        print("Failed to load image")
        return
        
    print("Testing exp.onnx on user validation image...")
    model_onnx = YOLO("models/exp.onnx")
    results_onnx = model_onnx(img, conf=0.01, verbose=False)
    boxes_onnx = results_onnx[0].boxes
    if boxes_onnx is not None and len(boxes_onnx) > 0:
        for box in boxes_onnx:
            print(f"exp.onnx -> Cls: {int(box.cls.item())}, Conf: {box.conf.item():.4f}")
    else:
        print("exp.onnx -> No detections.")

if __name__ == "__main__":
    main()
