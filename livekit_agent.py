import asyncio
import cv2
import numpy as np
import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, JobRequest, WorkerOptions, cli, AutoSubscribe
from src.attention import AttentionMonitor

load_dotenv()
logger = logging.getLogger("livekit-agent")

async def process_video_track(track: rtc.RemoteVideoTrack, ctx: JobContext, monitor: AttentionMonitor):
    logger.info(f"Subscribed to video track: {track.sid}")
    
    # 1. Prepare video source to publish back
    video_source = rtc.VideoSource(640, 480)
    local_track = rtc.LocalVideoTrack.create_video_track("ai-annotated", video_source)
    
    # 2. Publish AI track to the room
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.room.local_participant.publish_track(local_track, options)
    
    # 3. Read incoming frames
    video_stream = rtc.VideoStream(track)
    
    async for event in video_stream:
        frame = event.frame
        
        # Convert LiveKit VideoFrame (I420/RGBA) to Numpy BGR for OpenCV/YOLO
        # Get ARGB/RGBA representation
        argb_frame = frame.buffer.to_argb()
        
        # Create numpy array from raw bytes
        # ARGB format is technically packed. We can read it as 4 channels
        frame_data = bytearray(argb_frame.data)
        img_rgba = np.frombuffer(frame_data, dtype=np.uint8).reshape((argb_frame.height, argb_frame.width, 4))
        
        # Convert to BGR for YOLO
        # LiveKit's to_argb might actually be BGRA or RGBA in memory, usually RGBA.
        # cv2 handles it easily.
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        # Process the frame with YOLO
        annotated_img = monitor.process_frame(img_bgr)
        
        # Convert annotated BGR frame back to LiveKit RGBA VideoFrame
        img_rgba_out = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGBA)
        out_frame_data = img_rgba_out.tobytes()
        
        out_frame = rtc.VideoFrame(
            width=img_rgba_out.shape[1],
            height=img_rgba_out.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=out_frame_data
        )
        
        # Send it out
        video_source.capture_frame(out_frame)

async def entrypoint(ctx: JobContext):
    logger.info("Initializing Agent...")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    monitor = AttentionMonitor("models/phone-classification.pt")
    logger.info("Model loaded. Waiting for participants...")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(process_video_track(track, ctx, monitor))

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
