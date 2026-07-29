import cv2
import os
import uuid

def process_video_file(video_path, output_dir, monitor):
    """
    Reads a video, processes each frame with the monitor, and saves the output.
    Returns the path to the processed video.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
        
    output_filename = f"processed_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # We use mp4v codec for mp4 output, but for web playback, h264 is better.
    # OpenCV's default mp4v might not play in all browsers, so we might need ffmpeg or avc1
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Fallback to mp4v if avc1 is not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = monitor.process_frame(frame)
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    return output_path
