import urllib.request
import ssl
import cv2
from ultralytics import YOLO

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    url = "https://images.unsplash.com/photo-1511367461989-f85a21fda167?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&w=1200"
    image_path = "scratch/person_phone.jpg"
    try:
        urllib.request.urlretrieve(url, image_path)
        print("Downloaded test image.")
    except Exception as e:
        print("Failed to download:", e)
        return

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return
        
    print("Testing exp.pt...")
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
