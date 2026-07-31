import cv2

cap = cv2.VideoCapture("test.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height}, {fps} FPS, {frames} frames")
cap.release()
