import cv2
from src.inference.detection_pipeline import AttentionMonitor
import os
import uuid
import logging
import subprocess
import numpy as np

logger = logging.getLogger(__name__)


def _get_ffmpeg_path():
    """Get the ffmpeg binary path from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def process_video_file(video_path, output_dir, monitor, cancel_event=None):
    """
    Reads a video, processes each frame with the monitor, and saves the output.
    Uses ffmpeg (via imageio-ffmpeg) for H.264 encoding — browser-compatible.
    Returns the path to the processed video.
    """
    os.makedirs(output_dir, exist_ok=True)

    monitor.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or fps != fps:
        fps = 30.0

    output_filename = f"processed_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    ffmpeg_path = _get_ffmpeg_path()

    if ffmpeg_path:
        logger.info(f"Using ffmpeg for H.264 encoding: {ffmpeg_path}")
        _process_with_ffmpeg(cap, output_path, ffmpeg_path, fps, width, height, total_frames, monitor, cancel_event)
    else:
        logger.warning("ffmpeg not available, falling back to OpenCV mp4v codec")
        _process_with_opencv(cap, output_path, fps, width, height, total_frames, monitor, cancel_event)

    cap.release()
    return output_path


def _process_with_ffmpeg(cap, output_path, ffmpeg_path, fps, width, height, total_frames, monitor, cancel_event=None):
    """Process video frames and pipe them to ffmpeg for H.264 encoding."""
    # Ensure dimensions are even (H.264 requirement)
    out_width = width if width % 2 == 0 else width + 1
    out_height = height if height % 2 == 0 else height + 1

    cmd = [
        ffmpeg_path,
        '-y',                       # Overwrite output
        '-f', 'rawvideo',           # Input format
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',       # OpenCV uses BGR
        '-s', f'{out_width}x{out_height}',
        '-r', str(fps),
        '-i', '-',                  # Read from stdin
        '-c:v', 'libx264',         # H.264 codec
        '-preset', 'ultrafast',
        '-crf', '23',              # Quality (lower = better, 18-28 typical)
        '-pix_fmt', 'yuv420p',     # Browser-compatible pixel format
        '-movflags', '+faststart', # Enable streaming playback
        output_path
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_count = 0
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                logger.info("Video processing cancelled by user")
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = monitor.process_frame(frame)

            # Pad to even dimensions if needed
            if annotated_frame.shape[1] != out_width or annotated_frame.shape[0] != out_height:
                padded = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                padded[:annotated_frame.shape[0], :annotated_frame.shape[1]] = annotated_frame
                annotated_frame = padded

            process.stdin.write(annotated_frame.tobytes())
            frame_count += 1

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.0f}%)")
    finally:
        process.stdin.close()
        process.wait()

    if process.returncode != 0:
        logger.error("ffmpeg error: process returned non-zero exit code")
        raise RuntimeError("ffmpeg encoding failed")

    logger.info(f"Video processing complete: {frame_count} frames written to {output_path}")


def _process_with_opencv(cap, output_path, fps, width, height, total_frames, monitor, cancel_event=None):
    """Fallback: process video using OpenCV VideoWriter with mp4v codec."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError("Could not open VideoWriter with mp4v codec")

    frame_count = 0
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                logger.info("Video processing cancelled by user")
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            run_inference = (frame_count % 5 == 0)
            annotated_frame = monitor.process_frame(frame, run_inference=run_inference)
            out.write(annotated_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"Processed {frame_count}/{total_frames} frames ({progress:.0f}%)")
    finally:
        out.release()

    logger.info(f"Video processing complete: {frame_count} frames written to {output_path}")